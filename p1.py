import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def adjustIntensity (inImage, inRange=[], outRange = [0, 1]):
    fil,col = inImage.shape
    outImage= np.zeros((fil,col))
    if inRange == []:
        inRange = [np.amin(inImage), np.amax(inImage)]
    

    for i in range(fil):
        for j in range(col):
            outImage[i,j]= outRange[0]+ ((outRange[1]-outRange[0])*(inImage[i][j]-inRange[0]))/(inRange[1]-inRange[0])
    return outImage

def equalizeIntensity(inImage, nBins=256):
    histogram = [0] * nBins

    for pixel in inImage.flatten():
        histogram[pixel] += 1

    cumulativeHistogram = [sum(histogram[:i + 1]) for i in range(nBins)]

    cumulativeHistogram = [(x - min(cumulativeHistogram)) * 255 / (max(cumulativeHistogram) - min(cumulativeHistogram)) for x in cumulativeHistogram]

    equalizedImage = inImage.copy()
    for i in range(len(equalizedImage)):
        for j in range(len(equalizedImage[i])):
            equalizedImage[i][j] = round(cumulativeHistogram[inImage[i][j]])

    return equalizedImage



def sumaM(M1, M2):
    P, Q = M1.shape
    resultado = np.zeros((P,Q))
    for i in range(P):
        for j in range(Q):
            resultado[i][j]= M1[i][j] * M2[i][j]
    return np.sum(resultado)

def filterImage(inImage, kernel):
    if kernel.ndim == 1:
        kernel = np.expand_dims(kernel, axis=0)
    
    P, Q = kernel.shape
    IP, IQ = inImage.shape
    P2, Q2 = P // 2, Q // 2
    Pcenter, Qcenter = np.floor(P2) + 1, np.floor(Q2) + 1
    newIP, newQP = IP + 2 * (int(Pcenter) - 1), IQ + 2 * (int(Qcenter) - 1)
    inImageWP = np.zeros((newIP, newQP))
    inImageWP[int(Pcenter) - 1:int(Pcenter) - 1 + inImage.shape[0], int(Qcenter) - 1:int(Qcenter) - 1 + inImage.shape[1]] = inImage
    resultado = np.zeros((IP, IQ))
    
    for i in range(P2, (newIP - P2)):
        for j in range(Q2, (newQP - Q2)):
            
            # Recorremos kernel
            M1 = kernel
            M2 = inImageWP[i - P2:i + P2 + 1, j - Q2:j + Q2 + 1]
            resultado[i - P2][j - Q2] = sumaM(M1, M2)
            
    
    return resultado




#G(x) = (1/sqrt(2*pi*sigma)) * e ^(-(x^2) / (2*(sigma ^2)))
def gaussKernel1D(sigma):
    N = int(2 * np.ceil(3 * sigma)) + 1
    center = int(np.floor(N / 2))
    k = [0] * N  # Inicializa k como una lista de ceros

    for i in range(N):
        k[i] = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((i - center) ** 2) / (2 * (sigma ** 2)))

    #print(k)
    return np.array(k)

def gaussianFilter (inImage, sigma):#revisar
    kernel = gaussKernel1D(sigma)
    kernel = np.outer(kernel.reshape(-1, 1),kernel)
    return filterImage(inImage, kernel)


def mediana (matrix):
    sortedV = np.sort(matrix, axis = None)
    resultado = sortedV[int(np.ceil(sortedV.size/2) )]
    return resultado

def medianFilter (inImage, filterSize):
    IP,IQ = inImage.shape
    Pcenter,Qcenter = np.floor(filterSize/2) + 1 , np.floor(filterSize/2) + 1 
    newIP, newQP = IP + 2*(int(Pcenter)-1), IQ + 2*(int(Qcenter)-1)
    inImageWP = np.zeros((newIP, newQP))
    inImageWP[int(Pcenter)-1:int(Pcenter)-1+IP,int(Qcenter)-1:int(Qcenter)-1+IQ ] = inImage
    resultado = np.zeros((IP,IQ))
    for i in range (IP):
        for j in range (IQ):
            resultado[i][j] = mediana(inImageWP[i:i+filterSize,j:j+filterSize])
    return resultado

def compDilate (M1, M2, center):
    P, Q = M2.shape
    for i in range(P):
        for j in range(Q):
            if M2[i][ j] == 1 and M1[i] [j] == 255.0:  
                return 255.0
    return M1[int(center[0])][int(center[1])]


def dilate (inImage, SE, center=[]):
    #comprobar si inImage es binaria
    
    P,Q = inImage.shape
    if SE.ndim == 1:
        SE = np.expand_dims(SE, axis=0)
    SEP, SEQ = SE.shape
    if center == []:
        center = [np.floor(SEP/ 2) , np.floor(SEQ/2) ]
    P2, Q2 = SEP // 2, SEQ // 2
    Pcenter, Qcenter = np.floor(P2) + 1, np.floor(Q2) + 1
    newIP, newQP = P + 2 * (int(Pcenter) - 1), Q + 2 * (int(Qcenter) - 1)
    inImageWP = np.zeros((newIP, newQP))
    inImageWP[int(Pcenter) - 1:int(Pcenter) - 1 + inImage.shape[0], int(Qcenter) - 1:int(Qcenter) - 1 + inImage.shape[1]] = inImage
    
    resultado = np.zeros((P,Q))
    for i in range(P2, (newIP - P2)):
        for j in range(Q2, (newQP - Q2)):
            result = compDilate(inImageWP[i - P2:i + P2 + 1, j - Q2:j + Q2 + 1], SE, center)
            resultado[i - P2][j - Q2] = result

    return resultado



def compErode (M1, M2):
    P, Q = M2.shape
    for i in range(P):
        for j in range(Q):
            if M2[i][j] == 1 and M1[i][j] != 255.0:
                return 0
    return 255.0

def erode (inImage, SE, center=[]):
    P,Q = inImage.shape
    if SE.ndim == 1:
        SE = np.expand_dims(SE, axis=0)
    SEP, SEQ = SE.shape
    if center == []:
        center = [np.floor(SEP/ 2) , np.floor(SEQ/2) ]
    resultado = np.zeros((P,Q))
    for i in range(P - SEP + 1):
        for j in range(Q - SEQ + 1):
            result = compErode(inImage[i:i + SEP, j:j + SEQ], SE)
            resultado[i + int(center[0]), j + int(center[1])] = result

    return resultado    

def opening (inImage, SE, center=[]):
    return dilate(erode(inImage, SE, center), SE, center)

def closing (inImage, SE, center=[]):
    return erode(dilate(inImage, SE, center), SE, center)

def BtoW(inImage):
    P,Q = inImage.shape
    resultado = np.zeros((P,Q))
    for i in range(P):
        for j in range(Q):
            if inImage[i][j] == 1.0:
                resultado[i][j] = 0
            elif inImage[i][j] == 0:
                resultado[i][j] = 255.0
                
    
    return resultado

def convertirBooleano(matrix):
    return (matrix == 255.0)

def convertirNumerico(matrizBooleana):
    return np.where(matrizBooleana, 1.0, 0.0)

def hit_or_miss (inImage, objSEj, bgSE, center=[]):#revisar
    P,Q = bgSE.shape
    if center == []:
        center = [np.floor(P/ 2) , np.floor(Q/2) ]  
    HIT = erode(inImage, objSEj, center)
 
    MISS = erode(BtoW(inImage), bgSE, center)

    

    
    
    HIT = convertirBooleano(HIT)
    
    MISS = convertirBooleano(MISS)
    
    resultado = np.logical_and(HIT, MISS)
    
    resultado= convertirNumerico(resultado)
    
    return resultado
    

def gradientImage (inImage, operator):

    if operator == "Roberts":
        M1 = np.array([[-1, 0],
                      [0, 1]])
        M2 = np.rot90(M1, k=-1)
        
    elif operator == "CentralDiff":
        M1 = np.array([-1,0,1])
        M2 = np.reshape(M1, (-1, 1))
    elif operator == "Prewitt":
        M1 = np.array([[-1,0,1],
                      [-1,0,1],
                      [-1,0,1]])
        M2 = np.rot90(M1, k=-1)
        
    elif operator == "Sobel":
        M1 = np.array([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]])
        M2 = np.rot90(M1, k=-1)
        
    
    Gx = filterImage(inImage, M1)
    
    Gy = filterImage(inImage, M2)
    

    return [Gx, Gy]    


def LoG(inImage, sigma):
    N = 2 * np.ceil(3 * sigma) + 1
    center = int(np.floor(N / 2))
    resultado = np.zeros([int(N), int(N)])

    for i in range(int(N)):
        for j in range(int(N)):
            x, y = i - center, j - center
            r1 = x ** 2 + y ** 2 - 2*sigma ** 2
            r2 = 2 * np.pi * sigma ** 4
            r3 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            resultado[i, j] = r1 / r2 * r3

    # Aplicar el filtro LoG a la imagen de entrada utilizando convolución
    output = filterImage(inImage, resultado)

    output = (output - np.min(output)) / (np.max(output) - np.min(output)) * 255
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output




def supresion(magnitud, gx, gy):
    P,Q = magnitud.shape
    resultado = np.zeros((P,Q))
    for i in range(P-1):
        for j in range(Q - 1):
            artan = np.arctan2(gy[i][j],gx[i][j]) 
            artan = artan*180/np.pi
            p = False
            q = False
            if artan<=0 :
                artan += 180.0
            if (0 <= artan < 22.5) or (157.5 <= artan <= 180):
                p = magnitud[i][j] >= magnitud[i][j+1]
                q = magnitud[i][j] >= magnitud[i][j-1]
                
            elif (22.5 <= artan < 67.5):
                p = magnitud[i][j] >=magnitud[i+1][j-1]
                q = magnitud[i][j] >=magnitud[i-1][j+1]
            elif (67.5 <= artan < 112.5):
                p = magnitud[i][j] >=magnitud[i+1][j]
                q = magnitud[i][j] >=magnitud[i-1][j]
            elif (112.5 <= artan < 157.5):
                p = magnitud[i][j] >=magnitud[i-1][j-1]
                q = magnitud[i][j] >=magnitud[i+1][j+1]
            if p and q:
                resultado[i][j] = magnitud[i][j]
            else:
                resultado[i][j] = 0.0
    
    return resultado



def histeresis(suprimido, tlow, thigh):
    
    P , Q = suprimido.shape
    fuerte = np.zeros((P,Q))
    debil = np.zeros((P,Q))
    anterior = np.zeros((P,Q))
    for i in range(P):
        for j in range(Q):
            puntoActual = suprimido[i][j]
            if puntoActual >= tlow:
                debil[i][j] = 1
            if puntoActual >= thigh:
                fuerte[i][j] = 1
    debil = debil - fuerte
    while(not(np.array_equal(fuerte, anterior))):
        anterior = np.copy(fuerte)
        for i in range(P):
            for j in range(Q):
                if fuerte[i][j] == 1:
                    if debil[i-1][j-1] == 1:
                        fuerte[i-1][j-1] = 1
                    if debil[i][j-1] == 1:
                        fuerte[i][j-1] = 1
                    if debil[i+1][j-1] == 1:
                        fuerte[i+1][j-1] = 1
                    if debil[i-1][j] == 1:
                        fuerte[i-1][j] = 1
                    if debil[i+1][j] == 1:
                        fuerte[i+1][j] = 1
                    if debil[i-1][j+1] == 1:
                        fuerte[i-1][j+1] = 1
                    if debil[i][j+1] == 1:
                        fuerte[i][j+1] = 1
                    if debil[i+1][j+1] == 1:
                        fuerte[i+1][j+1] = 1
        
    return fuerte

 


def edgeCanny (inImage, sigma, tlow, thigh):#revisar
    resultado = gaussianFilter(inImage, sigma)
    
    gx, gy = gradientImage(resultado, "Sobel")
    magnitud = np.sqrt(np.power(gx,2) + np.power(gy,2))
    suprimido = supresion(magnitud, gx, gy)

    resultado = histeresis(suprimido,tlow, thigh)
      
    
    return resultado


def cornerSusan(inImage, r, t):
    P, Q = inImage.shape
    resultado = np.zeros((P, Q))
    for i in range(r, P - r):
        for j in range(r, Q - r):
            areaLocal = inImage[i - r:i + r + 1,j - r:j + r + 1]
            areaLocal = areaLocal - inImage[i][j]
            diferencia = np.abs(areaLocal)


            n = np.sum(diferencia < t)
            
            g = 0.75 * ((r * 2) ** 2)
            
            if n < g:
                resultado[i, j] = g - n#usanArea
    

    aux = np.ceil(np.unique(resultado).size/2)
    
    matriz = np.where(resultado>aux, resultado, 0)
    matriz = pasarBinario(matriz, 1)
    SE = np.array([
               [ 1, 0],
               [ 0, 1]])
    rojo =np.zeros(matriz.shape + (3,), dtype=np.uint8)
    matriz = opening(matriz, SE)
    
    return matriz, resultado






def pasarBinario(inImage, limite = 127):
    P,Q = inImage.shape
    resultado = np.zeros((P,Q))
    for i in range(P):
        for j in range(Q):
            if inImage[i][j] < limite:
                resultado[i][j] = 0.0
            else:
                resultado[i][j] = 255.0
    return resultado


if __name__ == "__main__":
    #adjustIntensity#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)  
    outImg=adjustIntensity(image,[],[0,127])
    cv2.imwrite('resultado.jpg', outImg)
    """
    #equalizehistogram#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    outImg = equalizeIntensity(image)
    cv2.imwrite('resultado.jpg', outImg)
    """
    #filterimage#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    kernel = 
    outImg = filterImage(image, kernel)
    outImg = adjustIntensity(outImg, [], [0,255])
    cv2.imwrite('resultado.jpg', outImg)
    """
    #gaussian filter#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    outImg = gaussianFilter(image, 0.7)
    cv2.imwrite('resultado.jpg', outImg)
    """
    #median filter#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    outImg = medianFilter(image, 7)
    cv2.imwrite('resultado.jpg', outImg)
    """
    #erode#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]])
    outImg = erode(image, kernel)
    cv2.imwrite('resultado.jpg', outImg)
    """
    #dilate#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]])
    outImg = dilate(image, kernel)
    cv2.imwrite('resultado.jpg', outImg)
    """
    #opening#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]])
    outImg = opening(image, kernel)
    cv2.imwrite('resultado.jpg', outImg)
    """
    #closing#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]])
    outImg = closing(image, kernel)
    cv2.imwrite('resultado.jpg', outImg)
    """
    #hitormiss#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    image = pasarBinario(image)
    SE = np.array([[0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0]])
    bgSE = np.array([[1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1]])
    outImg = hit_or_miss(image, SE,bgSE)
    outImg = adjustIntensity(outImg, [], [0,255])
    cv2.imwrite('resultado.jpg', outImg)
    """
    #gradientImage#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    gx,gy = gradientImage(image, "Prewitt")
    cv2.imwrite('resultadogx.jpg', gx)
    cv2.imwrite('resultadogy.jpg', gy)
    """
    #Log#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE) 
    outImg = LoG(image, 0.7)
    cv2.imwrite('resultado.jpg', outImg)
    """
    #edgeCanny#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE) 
    image = adjustIntensity(image)
    outImg = edgeCanny(image, 0.7, 0.1, 0.8)
    outImg = adjustIntensity(outImg, [], [0,255])
    cv2.imwrite('resultado.jpg', outImg)
    """
    #cornerSusan#
    """
    image = cv2.imread(IMAGEN, cv2.IMREAD_GRAYSCALE)
    
    outImg,outImg2 = cornerSusan(image, 7,200)
    print(np.unique(outImg))
    print(np.unique(outImg2))

    cv2.imwrite('resultado2.jpg', outImg2)
    cv2.imwrite('resultado.jpg', outImg)
    """