// Credits: whir-p3 (https://github.com/tcoratger/whir-p3) (MIT and Apache-2.0 licenses).

#[cfg(feature = "gpu")]
use cudarc::driver::safe::{CudaSlice, CudaStream, DevicePtr, DeviceSlice, SyncOnDrop};
#[cfg(feature = "gpu")]
use cudarc::driver::sys;
#[cfg(feature = "gpu")]
use std::sync::Arc;

mod commit;
pub use commit::*;
use poly::*;

mod open;
pub use open::*;

mod verify;
pub use verify::*;

mod dft;
pub use dft::*;

mod config;
pub use config::*;

mod merkle;
pub use merkle::DIGEST_ELEMS;
pub use merkle::WhirMerkleTree;
pub(crate) use merkle::*;

mod utils;
pub use utils::precompute_dft_twiddles;
// sample_ood_points is pub(crate) — users should inline it or access via the whir crate.
pub(crate) use utils::*;

mod matrix;
pub use matrix::DenseMatrix;
pub(crate) use matrix::*;

#[cfg(feature = "gpu")]
pub(crate) mod gpu_backend;
#[cfg(feature = "gpu")]
pub(crate) mod gpu_combine;
#[cfg(feature = "gpu")]
mod gpu_open;
#[cfg(feature = "gpu")]
pub(crate) mod gpu_prove;
#[cfg(feature = "gpu")]
pub use gpu_open::{
    GpuInitialOodData, GpuMerkleProverData, GpuTranscriptChunk, GpuTranscriptSeed, GpuWhirProverWorkspaces,
};

#[cfg(feature = "gpu")]
pub fn initialize_whir_gpu_backend() -> bool {
    gpu_backend::gpu().is_some()
}

#[cfg(feature = "gpu")]
pub fn initialize_whir_gpu_backend_on_stream(stream: Arc<CudaStream>) -> bool {
    gpu_backend::gpu_on_stream(stream).is_some()
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
pub enum GpuDeviceSlice {
    Owned(CudaSlice<u32>),
    Shared {
        data: Arc<CudaSlice<u32>>,
        offset_words: usize,
        len_words: usize,
    },
}

#[cfg(feature = "gpu")]
impl GpuDeviceSlice {
    pub fn shared(data: Arc<CudaSlice<u32>>, offset_words: usize, len_words: usize) -> Self {
        assert!(offset_words + len_words <= data.len());
        Self::Shared {
            data,
            offset_words,
            len_words,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            Self::Shared { len_words, .. } => *len_words,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(feature = "gpu")]
impl From<CudaSlice<u32>> for GpuDeviceSlice {
    fn from(data: CudaSlice<u32>) -> Self {
        Self::Owned(data)
    }
}

#[cfg(feature = "gpu")]
impl DeviceSlice<u32> for GpuDeviceSlice {
    fn len(&self) -> usize {
        GpuDeviceSlice::len(self)
    }

    fn stream(&self) -> &Arc<CudaStream> {
        match self {
            Self::Owned(data) => data.stream(),
            Self::Shared { data, .. } => data.stream(),
        }
    }
}

#[cfg(feature = "gpu")]
impl DevicePtr<u32> for GpuDeviceSlice {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        match self {
            Self::Owned(data) => data.device_ptr(stream),
            Self::Shared { data, offset_words, .. } => {
                let (ptr, guard) = data.device_ptr(stream);
                (ptr + (*offset_words as u64) * std::mem::size_of::<u32>() as u64, guard)
            }
        }
    }
}

#[cfg(feature = "gpu")]
pub struct GpuSparseStatement {
    pub total_num_variables: usize,
    pub point_len: usize,
    pub d_point_words: GpuDeviceSlice,
    pub values: Vec<GpuSparseValue>,
    pub is_next: bool,
}

#[cfg(feature = "gpu")]
pub struct GpuSparseValue {
    pub selector: usize,
    pub d_value: GpuDeviceSlice,
}

#[cfg(feature = "gpu")]
impl GpuSparseStatement {
    pub fn new<P>(total_num_variables: usize, point_len: usize, d_point_words: P, values: Vec<GpuSparseValue>) -> Self
    where
        P: Into<GpuDeviceSlice>,
    {
        let d_point_words = d_point_words.into();
        assert!(total_num_variables >= point_len);
        assert_eq!(d_point_words.len(), point_len * 5);
        Self {
            total_num_variables,
            point_len,
            d_point_words,
            values,
            is_next: false,
        }
    }

    pub fn new_next<P>(
        total_num_variables: usize,
        point_len: usize,
        d_point_words: P,
        values: Vec<GpuSparseValue>,
    ) -> Self
    where
        P: Into<GpuDeviceSlice>,
    {
        let d_point_words = d_point_words.into();
        assert!(total_num_variables >= point_len);
        assert_eq!(d_point_words.len(), point_len * 5);
        Self {
            total_num_variables,
            point_len,
            d_point_words,
            values,
            is_next: true,
        }
    }
}

#[cfg(feature = "gpu")]
impl GpuSparseValue {
    pub fn new<V>(selector: usize, d_value: V) -> Self
    where
        V: Into<GpuDeviceSlice>,
    {
        let d_value = d_value.into();
        assert_eq!(d_value.len(), 5);
        Self { selector, d_value }
    }

    pub fn shared(selector: usize, data: Arc<CudaSlice<u32>>, offset_words: usize, len_words: usize) -> Self {
        let d_value = GpuDeviceSlice::shared(data, offset_words, len_words);
        assert_eq!(d_value.len(), 5);
        Self { selector, d_value }
    }
}

#[derive(Clone, Debug)]
pub struct SparseStatement<EF> {
    pub total_num_variables: usize,
    pub point: MultilinearPoint<EF>,
    pub values: Vec<SparseValue<EF>>,
    /// When true, the weight polynomial is `next_mle(point, .)` instead of `eq(point, .)`.
    pub is_next: bool,
}

impl<EF> SparseStatement<EF> {
    pub fn new(total_num_variables: usize, point: MultilinearPoint<EF>, values: Vec<SparseValue<EF>>) -> Self {
        assert!(
            total_num_variables >= point.len(),
            "total_num_variables ({}) must be >= point.len() ({})",
            total_num_variables,
            point.len()
        );
        Self {
            total_num_variables,
            point,
            values,
            is_next: false,
        }
    }

    pub fn new_next(total_num_variables: usize, point: MultilinearPoint<EF>, values: Vec<SparseValue<EF>>) -> Self {
        assert!(
            total_num_variables >= point.len(),
            "total_num_variables ({}) must be >= point.len() ({})",
            total_num_variables,
            point.len()
        );
        Self {
            total_num_variables,
            point,
            values,
            is_next: true,
        }
    }

    pub fn unique_value(total_num_variables: usize, index: usize, value: EF) -> Self {
        Self {
            total_num_variables,
            point: MultilinearPoint(vec![]),
            values: vec![SparseValue { selector: index, value }],
            is_next: false,
        }
    }

    pub fn dense(point: MultilinearPoint<EF>, value: EF) -> Self {
        Self {
            total_num_variables: point.len(),
            point,
            values: vec![SparseValue { selector: 0, value }],
            is_next: false,
        }
    }

    pub fn selector_num_variables(&self) -> usize {
        self.total_num_variables
            .checked_sub(self.inner_num_variables())
            .expect("invariant violated: total_num_variables < point.len()")
    }

    pub fn inner_num_variables(&self) -> usize {
        self.point.len()
    }
}

#[derive(Clone, Debug)]
pub struct SparseValue<EF> {
    pub selector: usize,
    pub value: EF,
}

impl<EF> SparseValue<EF> {
    pub fn new(selector: usize, value: EF) -> Self {
        Self { selector, value }
    }
}
