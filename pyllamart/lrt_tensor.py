import struct
import torch

def dtype_to_lrt_dtype(dtype):
    if dtype == torch.float32:
        return 1
    else:
        raise Exception("dtype not supported")


def write_lrt_tensor(tensor: torch.Tensor, fp):
    ''' save the pytorch tensor to file with c8 format
    Args:
        tensor (pytorch.Tensor): input tensor
        fp (file like object): file stream to write
    '''
    assert tensor.dtype == torch.float32
    fp.write(b'TNSR')
    fp.write(struct.pack('<h', tensor.dim()))
    fp.write(struct.pack('<h', dtype_to_lrt_dtype(tensor.dtype)))
    for size in tensor.shape:
        fp.write(struct.pack('<i', size))
    for elem in tensor.reshape(-1):
        fp.write(struct.pack('<f', elem.item()))
    fp.write(struct.pack('<h', 0x55aa))

def write_tensor_dict(tensor_dict, filename):
    with open(filename, 'wb') as fp:
        fp.write(b'TDIC')
        fp.write(struct.pack('<i', len(tensor_dict)))
        for name, tensor in tensor_dict.items():
            if len(name) > 1024:
                raise Exception('name too long')
            name = name.encode('utf-8')
            fp.write(struct.pack('<h', len(name)))
            fp.write(name)
            write_lrt_tensor(tensor, fp)
        fp.write(struct.pack('<h', 0x55aa))
            

