from numpy.lib.stride_tricks import as_strided

def get_submatrices(layer,size,s=1):
      
      a=layer
      Hout = (a.shape[1] - size[0]) // s + 1
      Wout = (a.shape[2] - size[1]) // s + 1
      Stride = (a.strides[0], a.strides[1] * s, a.strides[2] * s, a.strides[1], a.strides[2], a.strides[3])
 
      a = as_strided(a, (a.shape[0], Hout, Wout, size[0], size[1], a.shape[3]), Stride)
      return a