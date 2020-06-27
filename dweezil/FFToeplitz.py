import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

class Circulant:
    def __init__(self,c_):
        self.len = len(c_)
        self.c = np.reshape(c_, self.len) #first column
        self.cVec = np.reshape(c_, (self.len,1))

    def asMatrix(self):
        A = self.c
        for n_ in range(1,self.len):
            n=self.len-n_
            x_shift = np.concatenate((self.c[n:],self.c[:n]))
            A = np.vstack([A, x_shift])
        return np.transpose(A)
    
    def matVec(self,x):
        #have noticed the FFT only works on flat vectors, so flatten these:
        xFlat = np.reshape(x, self.len)
        outFlat = np.fft.ifft(np.fft.fft(xFlat)*np.fft.fft(self.c))
        return outFlat.reshape(x.shape)

class BCCB:
    def __init__(self,cVecs_):
        self.cVecs = cVecs_
        self.blocks,self.sizes = cVecs_.shape

    def matVec(self,x_in):
        x = np.reshape(x_in,self.cVecs.shape)
        return np.reshape(np.fft.ifft2(np.fft.fft2(self.cVecs)*np.fft.fft2(x)),x_in.shape)
      
    def asMatrix(self):
        C = np.zeros((self.sizes*self.blocks,self.sizes*self.blocks), dtype=np.complex_)
        for n in range(self.blocks):
            for m in range(self.blocks):
                l = (n-m)%self.blocks
                C[self.sizes*n:(self.sizes*(n+1)),self.sizes*m:(self.sizes*(m+1))] = Circulant(self.cVecs[l]).asMatrix()
        return C

class PadBTTB:
    def __init__(self, Dd, Du, Dl, Ud, Uu, Ul, Ld, Lu, Ll, zR_,zL_):
        self.inBTTB = BTTB(Dd, Du, Dl, Ud, Uu, Ul, Ld, Lu, Ll)
        self.zR = zR_
        self.zL = zL_
        self.shape = (self.inBTTB.shape[0]-len(zL_),self.inBTTB.shape[1]-len(zR_))
        self.dtype = self.inBTTB.dtype

        # #define the zero padding operators
        # self.G = np.delete(np.eye(self.inBTTB.blockDim[1]*self.inBTTB.subBlockDim[1]),self.zR,1)
        # self.F = np.delete(np.eye(self.inBTTB.blockDim[0]*self.inBTTB.subBlockDim[0]),self.zL,0)

    def matVec(self,x):
        # xFlat = np.reshape(x, self.shape[1])
        # y = np.matmul(self.F,self.inBTTB.matVec(np.matmul(self.G,xFlat)))
        #possible more efficient to just manually delete the unwanted rows/columns here?
        #x_padded = np.insert(x,self.zR,0)
        non_zero_inds = np.delete(np.arange(self.inBTTB.shape[1]),self.zR)
        x_padded = np.zeros(self.inBTTB.shape[1],dtype='complex')
        x_padded[non_zero_inds] = x.ravel()
        return np.delete(self.inBTTB.matVec(x_padded),self.zL)
        # return y.reshape(self.shape[0])

    def asMatrix(self):
        A = self.inBTTB.asMatrix()
        Atrimed = np.delete(A,self.zR,1)
        Athinned = np.delete(Atrimed,self.zL,0)
        return Athinned

class BTTB:
    def __init__(self, Dd_, Du_, Dl_, Ud_, Uu_, Ul_, Ld_, Lu_, Ll_):
        self.Dd = Dd_
        self.Du = Du_
        self.Dl = Dl_
        self.Ud = Ud_
        self.Uu = Uu_
        self.Ul = Ul_
        self.Ld = Ld_
        self.Lu = Lu_
        self.Ll = Ll_
        self.blockDim = (len(Ld_)+1,len(Ud_)+1)
        self.subBlockDim = (len(self.Dl)+1,len(self.Du)+1)
        self.shape = (self.blockDim[0]*self.subBlockDim[0], self.blockDim[1]*self.subBlockDim[1])
        self.dtype = 'complex64'
        
        #now set up parameters to embed TBBT inside a BCCB
        self.circSubBlockDim = max(self.subBlockDim)*2
        if self.subBlockDim[1] > self.subBlockDim[0]:
            self.subBlockType = 'wide'
        elif self.subBlockDim[0] > self.subBlockDim[1]:
            self.subBlockType = 'tall'
        elif self.subBlockDim[0] == self.subBlockDim[0]:
            self.subBlockType = 'square'
        subBlockDiff = abs(self.subBlockDim[1] - self.subBlockDim[0])

        self.circBlockDim = max(self.blockDim)*2
        if self.blockDim[1] > self.blockDim[0]:
            self.blockType = 'wide'
        elif self.blockDim[1] < self.blockDim[0]:
            self.blockType = 'tall'
        elif self.blockDim[1] == self.blockDim[0]:
            self.blockType = 'square'
        blockDiff = abs(self.blockDim[1] - self.blockDim[0])

        #now create extended versions of input vectors for use in BCCB embedding
        Dux = Du_
        Dlx = Dl_
        Udx = Ud_
        Uux = Uu_
        Ulx = Ul_
        Ldx = Ld_
        Lux = Lu_
        Llx = Ll_

        # Fill leftover vector entries with zeros, until sub-blocks are square Toeplitz
        if self.subBlockType == 'wide':
            Ulx = np.hstack([self.Ul,np.zeros((self.blockDim[1]-1,subBlockDiff))])
            Llx = np.hstack([self.Ll,np.zeros((self.blockDim[0]-1,subBlockDiff))])
            Dlx = np.hstack([self.Dl,np.zeros(subBlockDiff)])
        elif self.subBlockType == 'tall':
            Uux = np.hstack([self.Uu,np.zeros((self.blockDim[1]-1,subBlockDiff))])
            Lux = np.hstack([self.Lu,np.zeros((self.blockDim[0]-1,subBlockDiff))])
            Dux = np.hstack([self.Du,np.zeros(subBlockDiff)])

        # Fill leftover blocks with zeros blocks, until full structure is block-Toeplitz
        if self.blockType == 'wide':
            Ldx = np.hstack([Ldx,np.zeros(blockDiff)])
            Lux = np.vstack([Lux,np.zeros((blockDiff,len(Lux[0])))])
            Llx = np.vstack([Llx,np.zeros((blockDiff,len(Llx[0])))])
        elif self.blockType == 'tall':
            Udx = np.hstack([Udx,np.zeros(blockDiff)])
            Uux = np.vstack([Uux,np.zeros((blockDiff,len(Uux[0])))])
            Ulx = np.vstack([Ulx,np.zeros((blockDiff,len(Ulx[0])))])
        
        self.cVecs = np.zeros((self.circBlockDim, self.circSubBlockDim), dtype=np.complex_)
        
        self.cVecs[0] = np.hstack([np.reshape(self.Dd,1), np.reshape(Dlx,len(Dlx)), 0, np.reshape(np.flipud(Dux),len(Dux))])
        ULblockSize = int(self.circBlockDim/2)-1
        for n in range(ULblockSize):
            self.cVecs[n+1] = np.hstack([np.reshape(Ldx[n],1), np.reshape(Llx[n],len(Llx[n])), 0, np.reshape(np.flipud(Lux[n]),len(Lux[n]))])
        for n in range(ULblockSize):
            n_ = self.circBlockDim - n - 1
            self.cVecs[n_] = np.hstack([np.reshape(Udx[n],1), np.reshape(Ulx[n],len(Ulx[n])), 0, np.reshape(np.flipud(Uux[n]),len(Uux[n]))])
                
        self.inBCCB = BCCB(self.cVecs)

        #now create the index vectors which can be used to extract the useful info from the matvec with the above BCCB monster we've just made
        self.lInds = []
        for j in range(self.blockDim[0]):
            self.lInds[(j*self.subBlockDim[0]):((j+1)*self.subBlockDim[0])] = [j*self.circSubBlockDim + x for x in range(self.subBlockDim[0])]        
        
        self.rInds = []#np.zeros(self.shape[1])
        for j in range(self.blockDim[1]):
            self.rInds[(j*self.subBlockDim[1]):((j+1)*self.subBlockDim[1])] = [j*self.circSubBlockDim + x for x in range(self.subBlockDim[1])]
    
    def asMatrix(self):
        N = self.blockDim[1]*self.subBlockDim[1]
        M = self.blockDim[0]*self.subBlockDim[0]
        A = np.array([]).reshape(0,N)
        for m in range(self.blockDim[0]):
            rowBlockTemp = np.array([]).reshape(self.subBlockDim[0],0)
            Umin = min(m,self.blockDim[1])
            for n in range(Umin):
                n_ = m - n - 1#Umin - n - 1
                rowBlockTemp = np.hstack([rowBlockTemp,wonkyToe(self.Ld[n_],self.Lu[n_],self.Ll[n_])])
            if m<=min(self.blockDim[0],self.blockDim[1])-1:
                rowBlockTemp = np.hstack([rowBlockTemp,wonkyToe(self.Dd,self.Du,self.Dl)])
            for n in range(self.blockDim[1]-Umin-1):
                rowBlockTemp = np.hstack([rowBlockTemp,wonkyToe(self.Ud[n],self.Uu[n],self.Ul[n])])
            A = np.vstack([A,rowBlockTemp])
        return A

    def matVec(self,x):
        xLong = np.zeros((self.circSubBlockDim*self.circBlockDim,1),dtype=complex)
        for n in range(len(self.rInds)):
            xLong[self.rInds[n]] = x[n]
        bLong = self.inBCCB.matVec(xLong)
##        print(xLong)
##        print(bLong)

        b = np.zeros((len(self.lInds),1),dtype=complex)
        for n in range(len(self.lInds)):
            b[n] = bLong[self.lInds[n]]
            
        return b
                
# class BTTBz():
#     def __init__(self,T,z):
#         self.T = T
#         self.zeroInds = z
#         self.shape = (T.shape[0]-len(z),T.shape[1]-len(z))
#         self.dtype = T.dtype
    
#         # self.G = np.delete(np.eye(self.T.ToetalLength),self.zeroInds,1)
#         # self.F = np.delete(np.eye(self.T.ToetalLength),self.zeroInds,0)

#     def matVec(self,x):
#         #xFlat = np.reshape(x, self.shape[0])
#         #y = np.matmul(self.F,self.T.matVec(np.matmul(self.G,xFlat)))
#         x_padded = np.insert(x,self.zeroInds,0)
#         return np.delete(self.T.matVec(x_padded),self.zeroInds)

class Toeplitz:
    def __init__(self,d_,u_,l_):
        n = len(l_)
        self.d = d_
        self.u = np.reshape(u_,n)
        self.l = np.reshape(l_,n)
        self.len = len(l_)+1
        self.circulantVec = np.concatenate(([self.d],self.l,[0],np.flipud(self.u)))

    def inCirculantMatrix(self):
        C = Circulant(self.circulantVec)
        return C.asMatrix()
    
    def matVec(self,x):
        xFlat = np.reshape(x, self.len)
        C = Circulant(self.circulantVec)
        zer0 = np.zeros(x.size)
        x0 = np.hstack([xFlat, zer0])
        Cx0 = C.matVec(x0)
        Tx = Cx0[:self.len]
        return Tx.reshape(x.shape)

    def asMatrix(self):
        A = [np.concatenate(([self.d],self.u))]
        lFlip = np.flipud(self.l)
        for n_ in range(1,self.len):
            n=self.len-n_-1
            x_shift = [np.concatenate((lFlip[n:],[self.d],self.u[:n]))]
            A = np.concatenate((A,x_shift))
        return np.array(A)

def wonkyToe(d,u,l):
    M = 1 + len(u)
    N = 1 + len(l)
    T = np.vstack([[d],np.reshape(l,(len(l),1))])
    uFlip = np.reshape(np.flipud(u),(len(u),1))
    for n in range(1,M):
        uMin = max(0,n-N)
        uFlip = np.reshape(np.flipud(u[uMin:n]),(len((u[uMin:n])),1))
        if n<N:
            d_ = np.array([[d]])
            lShift = np.reshape(l[:(N-n-1)],(N-n-1,1))
            col = np.vstack([uFlip,d_,lShift])
            col = np.vstack([uFlip,d_,lShift])
        else:
            col = uFlip
        T = np.hstack([T,col])
    return T

class BlockToeplitz:
    def __init__(self,D,U,L):
        N = len(D)
        self.numBlockLen = N
        self.blockLen = len(U[0][0])+1
        self.ToetalLength = self.numBlockLen*self.blockLen
        self.len = self.ToetalLength
        self.ToeBlock = np.empty((N,N), dtype=object)
        for n in range(N):
            for m in range(N):
                self.ToeBlock[n][m]=Toeplitz(D[n][m],U[n][m],L[n][m])

    def matVec(self,x):
        xFlat = np.reshape(x, self.len)
        b = np.transpose(np.zeros(x.size,dtype=np.complex_))
        M = self.blockLen
        for n in range(self.numBlockLen):
            for m in range(self.numBlockLen):
                b[n*M:(n+1)*M] += self.ToeBlock[n][m].matVec(xFlat[m*M:(m+1)*M])
        return np.reshape(b,x.shape)

    def asMatrix(self):
        A = np.array([]).reshape(0,self.ToetalLength)
        for n in range(self.numBlockLen):
            a = self.ToeBlock[n][0].asMatrix()
            for m in range(1,self.numBlockLen):
                a = np.hstack([a,self.ToeBlock[n][m].asMatrix()])
            A = np.vstack([A, a])
        return np.array(A)
                
class PaddedBlockToeplitz:
    def __init__(self,D,U,L,z):
        self.BlockToe = BlockToeplitz(D,U,L)
        self.zeroInds = z
        n = self.BlockToe.ToetalLength
        self.ToetalLengthWithoutPadding = n-len(z)
        self.len = self.ToetalLengthWithoutPadding
        self.dim = (n,n)

        #define the zero padding operators
        self.G = np.delete(np.eye(self.BlockToe.ToetalLength),self.zeroInds,1)
        self.F = np.delete(np.eye(self.BlockToe.ToetalLength),self.zeroInds,0)


    def matVec(self,x):
        xFlat = np.reshape(x, self.len)
        y = np.matmul(self.F,self.BlockToe.matVec(np.matmul(self.G,xFlat)))
        #possible more efficient to just manually delete the unwanted rows/columns here?
        return y.reshape(x.shape)

    def asMatrix(self):
        A = self.BlockToe.asMatrix()
        Atrimed = np.delete(A,self.zeroInds,1)
        Athinned = np.delete(Atrimed,self.zeroInds,0)
        return Athinned

# ------ stand-alone functions ------
def getLinearOperator(T):
    # for use with scipy's GMres, overloading the matrix input
    n = T.len
    def mv(v):
        outVect = T.matVec(v)
        return np.reshape(np.array(outVect),n)
    LT = LinearOperator((n,n), matvec=mv)
    return LT

def get_subcolumn(A,m_1,m_2,n):
    # A is a linear operator representing fast matvecs
    M,N = A.shape
    R = np.zeros([M,1])
    R[n] = 1
    r = (A @ R)[m_1:m_2,:]
    return r.reshape(m_2-m_1)

def get_entry(A,m,n):
    #the above function, just restricted to a single entry
    return get_subcolumn(A,m,m+1,n)
    
