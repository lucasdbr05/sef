use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    ops::ControlFlow,
    path::PathBuf,
};

use clap::{Parser, ValueEnum};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sef::chain::blk_file_reader::BlkFileReader;
use sef::chain::stream::for_each_epoch;
use sef::{
    chain::{error::ChainError, stream::EpochBatch},
    decoder::{self, BitcoinBlockVerifier},
    distribution::{AnyDistribution, BinomialDist, GeometricDist, IdealSoliton, PoissonDist, RobustSoliton},
    droplet::Encoder,
    epoch,
};

#[derive(ValueEnum, Clone, Debug)]
enum DistType {
    RobustSoliton,
    IdealSoliton,
    Poisson,
    Geometric,
    Binomial,
}

#[derive(Parser, Debug)]
#[command(name = "producer-consumer")]
struct Args {
    #[arg(
        long,
        default_value = "/Users/lucaslima/Library/Application Support/Bitcoin/signet_new/blocks"
    )]
    blocks_dir: PathBuf,

    #[arg(long, default_value_t = 500)]
    epoch_size: usize,

    #[arg(long, value_enum, default_value = "robust-soliton")]
    dist: DistType,

    #[arg(short, long, default_value_t = 0.3)]
    c: f64,

    #[arg(long, default_value_t = 0.01)]
    delta: f64,

    #[arg(long, default_value_t = 10)]
    buffer: usize,

    #[arg(long, default_value_t = 0)]
    max_droplets: usize,

    #[arg(long, default_value_t = 100)]
    trials: usize,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long, default_value = "producer_consumer_results_delta_001_c_03")]
    output_dir: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    fs::create_dir_all(&args.output_dir)?;

    let base_seed: u64 = match args.seed {
        Some(s) => s,
        None => ChaCha8Rng::from_entropy()
            .get_seed()
            .iter()
            .fold(0u64, |acc, &b| acc.wrapping_mul(257).wrapping_add(b as u64)),
    };

    println!(
        "blocks_dir={} | epoch_size={} | dist={:?} | c={} | delta={} | trials={} |",
        args.blocks_dir.display(),
        args.epoch_size,
        args.dist,
        args.c,
        args.delta,
        args.trials,
    );

    for trial in 0..args.trials {
        let trial_seed = base_seed
            .wrapping_add(trial as u64)
            .wrapping_mul(0x9E3779B97F4A7C15);
        let mut rng = ChaCha8Rng::seed_from_u64(trial_seed);

        let csv_path = args.output_dir.join(format!("trial_{:04}.csv", trial + 1));
        let mut csv = BufWriter::new(File::create(&csv_path)?);
        writeln!(csv, "epoch_idx,k,n_collected,decode_success,trial")?;

        let mut epoch_visitor = |batch: EpochBatch| -> Result<ControlFlow<()>, ChainError> {
            let k = batch.blocks.len();
            if k < 2 {
                return Ok(ControlFlow::Continue(()));
            }

            let first_hash = batch.blocks[0].hash.clone();
            let source_blocks: Vec<Vec<u8>> = batch.blocks.into_iter().map(|b| b.data).collect();

            let trusted_headers: Vec<bitcoin::block::Header> = source_blocks
                .iter()
                .enumerate()
                .map(|(i, data)| {
                    if data.len() < 80 {
                        Err(ChainError::Parse(format!(
                            "epoch {} block {}: tamanho < 80 bytes",
                            batch.index, i
                        )))
                    } else {
                        bitcoin::consensus::deserialize::<bitcoin::block::Header>(&data[..80])
                            .map_err(|e| {
                                ChainError::Parse(format!(
                                    "epoch {} block {}: header inválido: {}",
                                    batch.index, i, e
                                ))
                            })
                    }
                })
                .collect::<Result<_, _>>()?;

            let epoch_seed = epoch::compute_epoch_seed(batch.index, &first_hash);
            let max_droplets = if args.max_droplets == 0 {
                10 * k
            } else {
                args.max_droplets
            };

            let dist = match args.dist {
                DistType::RobustSoliton => AnyDistribution::RobustSoliton(RobustSoliton::new(k, args.c, args.delta)),
                DistType::IdealSoliton => AnyDistribution::IdealSoliton(IdealSoliton::new(k)),
                DistType::Poisson => AnyDistribution::Poisson(PoissonDist { k, lambda: 2.0 }),
                DistType::Geometric => AnyDistribution::Geometric(GeometricDist { k, p: 0.1 }),
                DistType::Binomial => AnyDistribution::Binomial(BinomialDist { k, n: k as u64 / 2, p: 0.5 }),
            };

            let params = sef::droplet::EpochParams::new(batch.index as u64, k as u32, epoch_seed);
            let encoder = Encoder::new(&params, &dist, &source_blocks);
            let verifier = BitcoinBlockVerifier {
                trusted_headers: trusted_headers.clone(),
            };

            let mut delivery_order: Vec<u64> = (0..max_droplets as u64).collect();
            delivery_order.shuffle(&mut rng);

            let mut collected: Vec<sef::droplet::Droplet> = Vec::new();
            let mut already_succeeded = false;

            for droplet_id in delivery_order {
                let droplet = encoder.generate(droplet_id);
                collected.push(droplet);

                let n_collected = collected.len();
                let result = decoder::peeling_decode(k, collected.clone(), &verifier);
                let ok = result.is_success();

                writeln!(
                    csv,
                    "{},{},{},{},{}",
                    batch.index,
                    k,
                    n_collected,
                    if ok { 1 } else { 0 },
                    trial + 1
                )?;

                if ok && !already_succeeded {
                    already_succeeded = true;
                    println!(
                        "  Trial {:4} | k={} | first success with {} droplets",
                        trial + 1,
                        k,
                        n_collected
                    );
                    break;
                }
            }

            Ok(ControlFlow::Continue(()))
        };

        let source = BlkFileReader::open(&args.blocks_dir)?;
        for_each_epoch(&source, args.epoch_size, args.buffer, &mut epoch_visitor)?;

        csv.flush()?;
        println!("Trial {:4}/{} finished", trial + 1, args.trials,);
    }

    Ok(())
}
