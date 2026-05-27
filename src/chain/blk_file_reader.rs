//! Backend for reading Bitcoin blocks directly from Bitcoin Core's `blk*.dat` files.
//!
//! Handles XOR obfuscation (introduced in Bitcoin Core v28+) by reading the
//! `xor.dat` key file from the blocks directory. Scans files in sorted order,
//! then chains blocks by `prev_blockhash` to produce canonical height ordering.
//!
//! For production use on mainnet, prefer
//! [`KernelBlockReader`](super::kernel_reader::KernelBlockReader) which uses
//! the validated block index and avoids the manual chain-ordering step.

use std::{
    collections::HashMap,
    fs, io,
    ops::ControlFlow,
    path::{Path, PathBuf},
};

use bitcoin::{BlockHash, block::Header, consensus::deserialize, hashes::Hash};

use crate::chain::{
    error::ChainError,
    stream::{BlockSource, RawBlock},
};

/// Reads blocks from Bitcoin Core's raw `blk*.dat` files.
///
/// Handles XOR obfuscation (Bitcoin Core v28+) and chains blocks by
/// `prev_blockhash` to produce canonical height ordering. For mainnet
/// production use, prefer [`KernelBlockReader`](super::kernel_reader::KernelBlockReader)
/// which uses the validated block index.
pub struct BlkFileReader {
    blocks_dir: PathBuf,
    xor_key: Vec<u8>,
}

impl BlkFileReader {
    /// Opens a blocks directory for reading.
    ///
    /// Reads the XOR key from `xor.dat` if present.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if `blocks_dir` cannot be read or `xor.dat`
    /// exists but is unreadable.
    pub fn open(blocks_dir: &Path) -> io::Result<Self> {
        let xor_path = blocks_dir.join("xor.dat");
        let xor_key = if xor_path.exists() {
            fs::read(&xor_path)?
        } else {
            vec![]
        };

        Ok(Self {
            blocks_dir: blocks_dir.to_path_buf(),
            xor_key,
        })
    }

    /// List all blk*.dat files in sorted order.
    fn blk_files(&self) -> io::Result<Vec<PathBuf>> {
        let mut files: Vec<PathBuf> = fs::read_dir(&self.blocks_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("blk") && n.ends_with(".dat"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        Ok(files)
    }

    /// De-obfuscate a full blk file in place.
    fn deobfuscate_file_in_place(&self, buf: &mut [u8]) {
        if self.xor_key.is_empty() {
            return;
        }
        let key_len = self.xor_key.len();
        for (idx, byte) in buf.iter_mut().enumerate() {
            *byte ^= self.xor_key[idx % key_len];
        }
    }

    /// Scan a single blk file, `emit` for each block found.
    fn scan_blk_file(
        &self,
        path: &Path,
        emit: &mut dyn FnMut(BlockHash, Vec<u8>) -> Result<ControlFlow<()>, ChainError>,
    ) -> Result<(), ChainError> {
        let mut file_bytes = fs::read(path)?;
        if file_bytes.len() < 8 {
            return Ok(());
        }
        self.deobfuscate_file_in_place(&mut file_bytes);

        let expected_magic = &file_bytes[0..4];
        let mut file_offset = 0usize;

        while file_offset + 8 <= file_bytes.len() {
            let header = &file_bytes[file_offset..file_offset + 8];
            let magic = &header[0..4];
            if magic == [0, 0, 0, 0] {
                break;
            }

            if magic != expected_magic {
                file_offset += 1;
                continue;
            }

            let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
            if size == 0 || size > 4_000_000 {
                break;
            }

            let data_offset = file_offset + 8;
            if data_offset + size > file_bytes.len() {
                break;
            }

            let block_data = file_bytes[data_offset..data_offset + size].to_vec();
            file_offset = data_offset + size;

            if block_data.len() >= 80 {
                match deserialize::<Header>(&block_data[..80]) {
                    Ok(header) => {
                        let hash = header.block_hash();
                        if let ControlFlow::Break(()) = emit(hash, block_data)? {
                            return Ok(());
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: failed to parse block header at offset {}: {}",
                            data_offset, e
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

struct BlockMeta {
    scan_order: usize,
    hash: BlockHash,
    prev_hash: BlockHash,
    data: Vec<u8>,
}

/// Two-pass strategy: first scans every `blk*.dat` file to collect all blocks
/// into memory, then walks the `prev_blockhash` chain from the earliest known
/// root to produce height-ordered [`RawBlock`]s. Bitcoin Core does not always
/// persist genesis in `blk*.dat` (notably on regtest), so the root may be the
/// first stored block whose parent is absent from the file set.
impl BlockSource for BlkFileReader {
    fn for_each_block(
        &self,
        visitor: &mut dyn FnMut(super::stream::RawBlock) -> Result<ControlFlow<()>, ChainError>,
    ) -> Result<(), ChainError> {
        let blk_files = self.blk_files()?;
        let mut blocks_by_hash: HashMap<BlockHash, BlockMeta> = HashMap::new();
        let mut genesis_hash: Option<BlockHash> = None;
        let mut next_scan_order = 0usize;

        for blk_path in &blk_files {
            self.scan_blk_file(blk_path, &mut |hash, data| {
                let header: Header =
                    deserialize(&data[..80]).map_err(|e| ChainError::Parse(e.to_string()))?;
                let prev = header.prev_blockhash;
                if prev == BlockHash::all_zeros() {
                    genesis_hash = Some(hash);
                }
                blocks_by_hash.insert(
                    hash,
                    BlockMeta {
                        scan_order: next_scan_order,
                        hash,
                        prev_hash: prev,
                        data,
                    },
                );
                next_scan_order += 1;
                Ok(ControlFlow::Continue(()))
            })?;
        }

        let mut child_of: HashMap<BlockHash, BlockHash> = HashMap::new();
        for meta in blocks_by_hash.values() {
            match child_of
                .get(&meta.prev_hash)
                .and_then(|hash| blocks_by_hash.get(hash))
            {
                Some(existing) if existing.scan_order <= meta.scan_order => {}
                _ => {
                    child_of.insert(meta.prev_hash, meta.hash);
                }
            }
        }

        let root_hash = genesis_hash
            .or_else(|| {
                blocks_by_hash
                    .values()
                    .filter(|meta| !blocks_by_hash.contains_key(&meta.prev_hash))
                    .min_by_key(|meta| meta.scan_order)
                    .map(|meta| meta.hash)
            })
            .ok_or_else(|| ChainError::Parse("no chain root found".into()))?;
        let mut current = root_hash;
        let mut height = 0;

        while let Some(m) = blocks_by_hash.remove(&current) {
            let block = RawBlock {
                height,
                hash: m.hash.to_string(),
                data: m.data,
            };

            if let ControlFlow::Break(()) = visitor(block)? {
                return Ok(());
            }

            height += 1;
            match child_of.get(&current) {
                Some(&next) => current = next,
                None => break,
            }
        }
        Ok(())
    }
}
