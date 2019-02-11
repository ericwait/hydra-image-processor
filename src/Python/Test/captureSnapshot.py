# -*- coding: utf-8 -*-
import platform
import subprocess

import numpy as np
import scipy.io

from os import path, makedirs
from datetime import datetime

import HIP


def makePlatformStr():
    return platform.system() + '_' + platform.machine();


def loadGitCommit(rootDir):
    cProc = subprocess.run(['git','-C',rootDir,'rev-parse','HEAD'], capture_output=True, text=True)
    if cProc.returncode != 0:
        return datetime.today().strftime('%Y-%m-%d-%H%M')
    else:
        return cProc.stdout.strip()

def convertType(im, outType):
    if outType == 'bool':
        return (im >= 0.5)
    elif outType == 'uint8':
        return (np.round((2**8-1)*im)).astype(outType)
    elif outType == 'uint16':
        return (np.round((2**14-1)*im)).astype(outType)
    elif outType == 'int16':
        return (np.round((2**14-1)*(im-0.5))).astype(outType)
    elif outType == 'uint32':
        return (np.round((2**20-1)*im)).astype(outType)
    elif outType == 'int32':
        return (np.round((2**20-1)*(im-0.5))).astype(outType)
    elif outType == 'float':
        return im.astype(outType)
    elif outType == 'double':
        return im
    return None


def saveCmdOutput(outDir, imOut, cmdStr, numdims, dataType):
    outDataType = dataType
    if dataType == 'float':
        outDataType = 'single'

    outFile = path.join(outDir, cmdStr+'_'+str(numdims)+'d_'+str(outDataType)+'.mat')
    scipy.io.savemat(outFile, {'imOut': imOut})


def runAllCommands(outDir, imRect, imNoise, imSum, numdims, dataType):
    pxSize = np.array([1.0,1.0,3.0])
    numsdims = min([3,numdims])
    
    fullkerndims = [5,5,3]
    kerndims = fullkerndims[slice(numsdims)]
    
    # TODO: Try this with order='C' as well
    kernel = np.ones(kerndims, dtype=np.float, order='F')    
    sigmas = np.concatenate((10 / pxSize[range(numsdims)], np.zeros(3-numsdims)))
    highpassSigmas = 2*sigmas
    
    imOut = HIP.Closure(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'Closure', numdims, dataType)

    imOut = HIP.ElementWiseDifference(imSum, imNoise)
    saveCmdOutput(outDir, imOut, 'ElementWiseDifference', numdims, dataType)

    imOut = HIP.EntropyFilter(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'EntropyFilter', numdims, dataType)

    imOut = HIP.Gaussian(imSum, sigmas)
    saveCmdOutput(outDir, imOut, 'Gaussian', numdims, dataType)

    # TODO: Min/Max is caussing a crash in the next kernel
#    imMin,imMax = HIP.GetMinMax(imSum)
#    imOut = np.array([imMin,imMax])
#    saveCmdOutput(outDir, imOut, 'GetMinMax', numdims, dataType)

    # imOut = HIP.HighPassFilter(imSum, highpassSigmas)
    # saveCmdOutput(outDir, imOut, 'HighPassFilter', numdims, dataType)

    imOut = HIP.LoG(imSum, sigmas)
    saveCmdOutput(outDir, imOut, 'LoG', numdims, dataType)

    imOut = HIP.MaxFilter(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'MaxFilter', numdims, dataType)

    imOut = HIP.MeanFilter(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'MeanFilter', numdims, dataType)

    imOut = HIP.MedianFilter(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'MedianFilter', numdims, dataType)

    imOut = HIP.MinFilter(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'MinFilter', numdims, dataType)

    imOut = HIP.MultiplySum(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'MultiplySum', numdims, dataType)

    imOut = HIP.Opener(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'Opener', numdims, dataType)

    imOut = HIP.StdFilter(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'StdFilter', numdims, dataType)

    sumOut = HIP.Sum(imSum)
    imOut = np.array([sumOut])
    saveCmdOutput(outDir, imOut, 'Sum', numdims, dataType)

    imOut = HIP.VarFilter(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'VarFilter', numdims, dataType)

    imOut = HIP.WienerFilter(imSum, kernel)
    saveCmdOutput(outDir, imOut, 'WienerFilter', numdims, dataType)



def captureSnapshot(rootDir):
    testDir = path.join(rootDir, 'Testing')
    platStr = makePlatformStr()
    revStr = loadGitCommit(rootDir)
    
    snapName = 'snapshot_python_' + platStr + '_' + revStr
    snapDir = path.join(testDir, snapName)
    
    makedirs(snapDir, exist_ok=True)

    # TODO: Validate type conversions    
    data_types = ['bool', 'uint8', 'uint16', 'int16', 'uint32', 'int32', 'float', 'double']

    for d in range(2,6):
        rectmat = scipy.io.loadmat(path.join(testDir, 'Images', 'test_image_'+str(d)+'d_rect.mat'), matlab_compatible=True)
        noisemat = scipy.io.loadmat(path.join(testDir, 'Images', 'test_image_'+str(d)+'d_noise.mat'), matlab_compatible=True)
        summat = scipy.io.loadmat(path.join(testDir, 'Images', 'test_image_'+str(d)+'d_sum.mat'), matlab_compatible=True)
        
        for i in range(len(data_types)):
            imRect = convertType(rectmat['im'], data_types[i])
            imNoise = convertType(noisemat['im'], data_types[i])
            imSum = convertType(summat['im'], data_types[i])
            
            runAllCommands(snapDir, imRect, imNoise, imSum, d, data_types[i])
        