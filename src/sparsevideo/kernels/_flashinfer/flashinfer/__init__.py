"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

try:
    from ._build_meta import __version__ as __version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"

# SparseVideo vendored subset: only the APIs we actually use.
from . import jit as jit
from .cascade import merge_state as merge_state
from .cascade import merge_state_in_place as merge_state_in_place
from .cascade import merge_states as merge_states
from .prefill import single_prefill_with_kv_cache as single_prefill_with_kv_cache
from .prefill import (
    single_prefill_with_kv_cache_return_lse as single_prefill_with_kv_cache_return_lse,
)
from .sparse import BlockSparseAttentionWrapper as BlockSparseAttentionWrapper
from .sparse import (
    VariableBlockSparseAttentionWrapper as VariableBlockSparseAttentionWrapper,
)
