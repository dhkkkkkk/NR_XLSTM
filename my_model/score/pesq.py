from pesq import pesq

def pesq_score(self, audios, rate):
    if len(audios) != 2:
        raise ValueError('PESQ needs a reference and a test signals.')
        return None
    return pesq(rate, audios[1], audios[0], 'wb')