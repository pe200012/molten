module Molten.Reference
  ( MatrixGemmRef(..)
  , ReferenceOutput(..)
  , axpyVectorRef
  , dotVectorRef
  , gemmMatrixRef
  , mapArrayRef
  , reduceAllArrayRef
  , reshapeArrayRef
  , runProgramCpu
  , zipWithArrayRef
  ) where

import Molten.Array.Program (ReferenceOutput(..), runProgramCpu)
import Molten.Internal.Reference.Array
  ( mapArrayRef
  , reduceAllArrayRef
  , reshapeArrayRef
  , zipWithArrayRef
  )
import Molten.Internal.Reference.BLAS
  ( MatrixGemmRef(..)
  , axpyVectorRef
  , dotVectorRef
  , gemmMatrixRef
  )
