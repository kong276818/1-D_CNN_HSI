import numpy as np
from bitstring import ConstBitStream as CBS

class SviParser:
    def __init__(self, file):
        self.file = file

    def parse(self):
        self.f = open(self.file, "rb")
        data = CBS(self.f.read(128))
        sigmsg = data.read('bytes:4')
        if sigmsg == b'.SSI':
            print("> Read the Hyper-Spectral/ Image file. -> ", self.file)
        elif sigmsg == b'.SRI':
            print("  Read the Color Image file.")

        version = data.read('uintle:16')
        print('  Version: ', version)
        data.read('bytes:26')

        self.nSpacial = data.read('uintle:32')
        self.nSpectral = data.read('uintle:32')
        self.nPixelFormat = data.read('uintle:32')
        self.nOffsetX = data.read('uintle:16')
        self.nOffsetY = data.read('uintle:16')
        self.bCalibrated = data.read('uintle:16')
        self.nBinningH = data.read('uintle:16')
        self.nBinningV = data.read('uintle:16')

        print("  Spacial Width: ", self.nSpacial)
        print("  Spectral Width: ", self.nSpectral)
        print("  Pixel Format: ", self.nPixelFormat)
        print("  Offset-X: ", self.nOffsetX)
        print("  Offset-Y: ", self.nOffsetY)
        print("  Calibrated: ", self.bCalibrated)
        print("  Binning H: ", self.nBinningH)
        print("  Binning V: ", self.nBinningV)

        self.nImageSize = self.nSpacial * self.nSpectral
        data = np.fromfile(self.f, dtype=np.uint16)
        data = data[:int(len(data) / self.nImageSize) * self.nImageSize]
        self.images = data.reshape(-1, self.nSpectral, self.nSpacial).astype(np.int16)
        print("  Shape: ", self.images.shape)



