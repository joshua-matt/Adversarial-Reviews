import numpy as np
import numpy.random as rn
import time

def gen_random(n): # Random binary list
    return rn.randint(0,2,n)

def gen_honest_reviews(GT, mh, p): # Generates 'mh' honest reviews, where each review is correct on any particular with probability p
    return [np.array([it if rn.rand() < p else 1-it for it in GT]) for i in range(int(mh))]

def gen_mal_reviews(GT, Rh, mm, n, setting="z", p_=0): # Generates 'mm' malicious reviews, with four types to choose from
    mm = int(np.floor(mm))
    Ra = []

    for i in range(mm):
        c = setting[i%len(setting)] # Generates next review according to 'setting' string
        if c == 'i':
            Ra.append(np.array([1-it for it in Rh[rn.randint(len(Rh))]])) # Adds the inversion of a random honest review
        elif c == 'r':
            Ra.append(gen_random(n)) # Random review
        elif c == 'p':
            Ra.append(gen_honest_reviews(GT, 1, p_)[0]) # Flips each entry of GT with probability p_
        else:
            Ra.append(np.zeros(n)) # All zeros

    return Ra

def hamm_dist(r1, r2): # Hamming distance between two reviews
    return np.sum(np.abs(r1-r2)) # (r1-r2)[i] = 1 iff r1[i] != r2[i]

def estimate(R, ep, p, alpha):
    q = 1 - p
    m = len(R)
    n = len(R[0])

    t1 = time.time()

    ##### STEP 1: REMOVE REVIEWS TOO FAR FROM THE REST #####
    neighbors = {i:1 for i in range(m)} # Number of reviews that each review is "close" to

    for i in range(m):
        ri = R[i]
        for j in range(i+1, m):
            rj = R[j]
            if hamm_dist(ri, rj) <= (1+ep)*2*p*q*n:
                neighbors[i] += 1
                neighbors[j] += 1

    R1 = [i for i in neighbors if neighbors[i] >= (1-alpha) * m] # Only take reviews that have enough neighbors

    if debugging:
        print("Step 1 took %.3f seconds." % (time.time() - t1))

    t2 = time.time()

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

    if debugging:
        print("Step 2 took %.3f seconds." % (time.time() - t2))

    R2 = [i for i in R1 if i not in marked2]

    if len(R2) == 0:
        raise ZeroDivisionError("The final set of reviews is empty. Increase your epsilon!")

    theta_hat = np.int32(np.round(np.sum(np.array([R[i] for i in R2]), axis=0) / len(R2)))

    return theta_hat

debugging = True

def average_scores(M, iters): # See how performance varies with m
    n = 200

    alpha = 0.25
    p = 0.75
    ep = 0.142

    totals = {m:0 for m in M}

    for m in M:
        for i in range(iters):
            GT = gen_random(n)

            Rh = gen_honest_reviews(GT, (1 - alpha) * m, p)
            Ra = gen_mal_reviews(GT, Rh, alpha * m, n, setting="pert", p=1-p)

            R = Rh + Ra
            rn.shuffle(R)

            score = (1 - (hamm_dist(estimate(R, ep, p, alpha), GT) / n)) * 100
            totals[m] += score
    return {m:totals[m]/iters for m in M}


def main():
    n = 200
    m = 100

    alpha = 0.25
    p = 0.75
    ep = 0.12

    GT = gen_random(n)

    Rh = gen_honest_reviews(GT, (1-alpha)*m, p)
    Ra = gen_mal_reviews(GT, Rh, alpha*m, n, setting="pr", p=1-p)

    R = Rh + Ra
    rn.shuffle(R)

    t1 = time.time()

    score = (1 - (hamm_dist(estimate(R, ep, p, alpha), GT) / n))*100

    if not debugging:
        print("___________")
        print("| n: %d" % (n))
        print("| m: %d" % (m))
        print("| p: %.3f" % (p))
        print("| α: %.3f" % (alpha))
        print("| ε: %.3f" % (ep))
    print("SCORE: %.1f%%. Executed in %.3f seconds." % (score, time.time() - t1))
#print(average_scores(list(range(100,600,100)), 8))
main()
