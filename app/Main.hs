{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import Control.Monad (unless)
import qualified Data.Vector.Storable as VS
import Molten
  ( Context
  , DeviceId(..)
  , Gemm(..)
  , Transpose(..)
  , axpy
  , copyD2H
  , copyH2D
  , dot
  , gemm
  , readHostBufferToVector
  , withContext
  , withDeviceBuffer
  , withHostBuffer
  , withHostBufferFromVector
  )
import ROCm.HIP.Device
  ( hipGetCurrentDeviceGcnArchName
  , hipGetCurrentDeviceName
  , hipGetDeviceCount
  )
import System.Environment (getArgs)

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["--probe-context"] -> do
      putStrLn "Molten demo running in context probe mode."
      withContext (DeviceId 0) (const (putStrLn "Molten context probe: OK"))
    _ -> runDemoMain

runDemoMain :: IO ()
runDemoMain = do
  deviceCount <- hipGetDeviceCount
  if deviceCount <= 0
    then putStrLn "Molten demo: no ROCm GPU available; skipped."
    else do
      deviceName <- hipGetCurrentDeviceName
      archName <- hipGetCurrentDeviceGcnArchName
      putStrLn ("Molten demo running on: " <> deviceName <> " (" <> archName <> ")")
      withContext (DeviceId 0) runDemo

runDemo :: Context -> IO ()
runDemo ctx = do
  runTransferDemo ctx
  runBlasDemo ctx

runTransferDemo :: Context -> IO ()
runTransferDemo ctx = do
  let input = VS.fromList [0 .. 15 :: Float]
      n = VS.length input
  withHostBufferFromVector input $ \hostIn ->
    withDeviceBuffer @Float ctx n $ \deviceBuffer ->
      withHostBuffer @Float n $ \hostOut -> do
        copyH2D ctx deviceBuffer hostIn
        copyD2H ctx hostOut deviceBuffer
        output <- readHostBufferToVector hostOut
        unless (output == input) $
          fail ("Molten transfer demo mismatch: expected " <> show input <> ", got " <> show output)
  putStrLn "Molten demo: transfer roundtrip OK"

runBlasDemo :: Context -> IO ()
runBlasDemo ctx = do
  let x = VS.fromList [1, 2, 3, 4 :: Float]
      y = VS.fromList [10, 20, 30, 40 :: Float]
      expectedAxpy = VS.fromList [12, 24, 36, 48 :: Float]
      expectedDot = 360.0 :: Float
      a = VS.fromList [1, 2, 3, 4 :: Float]
      b = VS.fromList [5, 6, 7, 8 :: Float]
      c0 = VS.fromList [0, 0, 0, 0 :: Float]
      expectedGemm = VS.fromList [19, 22, 43, 50 :: Float]

  withHostBufferFromVector x $ \hostX ->
    withHostBufferFromVector y $ \hostY ->
      withDeviceBuffer @Float ctx 4 $ \deviceX ->
        withDeviceBuffer @Float ctx 4 $ \deviceY ->
          withHostBuffer @Float 4 $ \hostYOut -> do
            copyH2D ctx deviceX hostX
            copyH2D ctx deviceY hostY
            axpy ctx 2.0 deviceX deviceY
            copyD2H ctx hostYOut deviceY
            yOut <- readHostBufferToVector hostYOut
            unless (approxVector yOut expectedAxpy) $
              fail ("Molten BLAS axpy mismatch: expected " <> show expectedAxpy <> ", got " <> show yOut)
            dotResult <- dot ctx deviceX deviceY
            unless (approxFloat dotResult expectedDot) $
              fail ("Molten BLAS dot mismatch: expected " <> show expectedDot <> ", got " <> show dotResult)

  withHostBufferFromVector a $ \hostA ->
    withHostBufferFromVector b $ \hostB ->
      withHostBufferFromVector c0 $ \hostC ->
        withDeviceBuffer @Float ctx 4 $ \deviceA ->
          withDeviceBuffer @Float ctx 4 $ \deviceB ->
            withDeviceBuffer @Float ctx 4 $ \deviceC ->
              withHostBuffer @Float 4 $ \hostCOut -> do
                copyH2D ctx deviceA hostA
                copyH2D ctx deviceB hostB
                copyH2D ctx deviceC hostC
                gemm
                  ctx
                  Gemm
                    { gemmTransA = NoTranspose
                    , gemmTransB = NoTranspose
                    , gemmM = 2
                    , gemmN = 2
                    , gemmK = 2
                    , gemmAlpha = 1.0
                    , gemmA = deviceA
                    , gemmLda = 2
                    , gemmB = deviceB
                    , gemmLdb = 2
                    , gemmBeta = 0.0
                    , gemmC = deviceC
                    , gemmLdc = 2
                    }
                copyD2H ctx hostCOut deviceC
                cOut <- readHostBufferToVector hostCOut
                unless (approxVector cOut expectedGemm) $
                  fail ("Molten BLAS gemm mismatch: expected " <> show expectedGemm <> ", got " <> show cOut)

  putStrLn "Molten demo: axpy OK"
  putStrLn "Molten demo: dot OK"
  putStrLn "Molten demo: gemm OK"

approxVector :: VS.Vector Float -> VS.Vector Float -> Bool
approxVector left right =
  VS.length left == VS.length right
    && and (VS.toList (VS.zipWith approxFloat left right))

approxFloat :: Float -> Float -> Bool
approxFloat left right = abs (left - right) <= 1.0e-4
