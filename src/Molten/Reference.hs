module Molten.Reference
  ( MatrixGemmRef(..)
  , ReferenceOutput(..)
  , axpyVectorRef
  , broadcastColsRef
  , broadcastRowsRef
  , dotVectorRef
  , gemmMatrixRef
  , mapArrayRef
  , maxColsRef
  , maxRowsRef
  , reduceAllArrayRef
  , reshapeArrayRef
  , runProgramCpu
  , sumColsRef
  , sumRowsRef
  , zipWithArrayRef
  ) where

import Molten.Array.Program (ReferenceOutput(..), runProgramCpu)
import Molten.Internal.Reference.Array
  ( mapArrayRef
  , reduceAllArrayRef
  , reshapeArrayRef
  , zipWithArrayRef
  )
import Molten.Internal.Reference.Axis2D
  ( broadcastColsRef
  , broadcastRowsRef
  , maxColsRef
  , maxRowsRef
  , sumColsRef
  , sumRowsRef
  )
import Molten.Internal.Reference.BLAS
  ( MatrixGemmRef(..)
  , axpyVectorRef
  , dotVectorRef
  , gemmMatrixRef
  )
