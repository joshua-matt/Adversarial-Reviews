import numpy as np

def gen_random(n): # Random binary list
    return np.random.randint(0,2,n)

def gen_honest_reviews(GT, mh, p): # Generates 'mh' honest reviews, where each review is correct on any particular with probability p
    return [np.array([it if np.random.rand() < p else 1-it for it in GT]) for i in range(int(mh))]

def gen_mal_reviews(gt, Rh, mm, n, setting="zero"): # Generates 'mm' malicious reviews, with options of
                                                    # a) inverting the honest reviews, b) creating random reviews, or c) creating reviews of all 0
    mm = int(np.floor(mm))
    if setting == "invert":
        return [np.array([1-it for it in Rh[i]]) for i in range(len(Rh))] + [gen_random(n) for i in range(m-len(Rh))]
    elif setting == "random":
        return [gen_random(n) for i in range(mm)]
    else:
        return [np.zeros(n) for i in range(mm)]

def hamm_dist(r1, r2): # Hamming distance between two reviews
    s = 0
    for i in range(len(r1)):
        if r1[i] != r2[i]:
            s += 1
    return s

def estimate(R, ep, p, alpha):
    q = 1 - p
    m = len(R)
    n = len(R[0])

    ##### STEP 1: REMOVE REVIEWS TOO FAR FROM OTHERS #####
    marked1 = set()
    R1 = []

    for i in range(m):
        if i in marked1:
            continue
        ri = R[i]
        neighs = 0
        for j in range(m):
            if j in marked1:
                continue
            rj = R[j]
            if hamm_dist(ri, rj) <= (1+ep)*2*p*q*n:
                neighs += 1
        if neighs >= (1-alpha)*m:
            R1.append(i)
        else:
            marked1.add(i)

    ##### STEP 2: REMOVE REVIEWS TOO CLOSE TO OTHERS #####
    marked2 = set()
    for i in range(len(R1)):
        if i in marked2:
            continue
        ri = R[R1[i]]
        for j in range(i+1, len(R1)):
            if j in marked2:
                continue
            rj = R[R1[j]]
            if hamm_dist(ri, rj) < (1-ep)*2*p*q*n:
                marked2.add(i)
                marked2.add(j)

    R2 = [i for i in R1 if i not in marked2]

    if len(R2) == 0:
        raise ZeroDivisionError("The final set of reviews is empty. Increase your epsilon!")

    theta_hat = np.int32(np.round(np.sum(np.array([R[i] for i in R2]), axis=0) / len(R2)))

    return theta_hat

n = 200
m = 100

alpha = 0.25
p = 0.75
ep = 0.7

GT = gen_random(n)

Rh = gen_honest_reviews(GT, (1-alpha)*m, p)
Ra = gen_mal_reviews(GT, Rh, alpha*m, n, setting="invert")

R = Rh + Ra

print(hamm_dist(estimate(R, ep, p, alpha), GT) / n)
#print(estimate(R, ep, p, alpha))
