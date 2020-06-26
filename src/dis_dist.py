from dis_gauss import discretegauss

def dl_gauss(sigma2=1.0, L=1.0):
    return L * sample_dgauss(sigma2*L*L)
