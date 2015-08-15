


load_mnist <- function() {
  
  load_image_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n = n * nrow * ncol, size = 1, signed = F)
    x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    x
  }
  
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  
  train <<- cbind(load_label_file(filename = 'mnist/train-labels.idx1-ubyte'), 
                  load_image_file(filename = 'mnist/train-images.idx3-ubyte')  / 255)
  
  test <<- cbind(load_label_file(filename = 'mnist/t10k-labels.idx1-ubyte'), 
                 load_image_file(filename = 'mnist/t10k-images.idx3-ubyte')  / 255)
  
  
}



