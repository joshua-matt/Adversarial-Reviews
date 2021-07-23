import numpy as np
import numpy.random as rn
import time

def gen_random(n): # Random binary list
    return rn.randint(0,2,n)

def gen_honest_reviews(GT, mh, p, Rh_init): # Generates 'mh' honest reviews, where each review is correct on any particular with probability p
    Rh = np.copy(Rh_init)

    n = len(GT)
    i = -1

    while 1:
        i += rn.geometric(1-p)
        if i >= len(Rh):
            break
        Rh[i] = 1 - Rh[i]

    return Rh.reshape((int(mh), n))

def gen_mal_reviews(GT, Rh, mm, n, setting="z", p=0): # Generates 'mm' malicious reviews, with options of
                                                    # a) inverting the honest reviews, b) creating random reviews, or c) creating reviews of all 0
    mm = int(np.floor(mm))
    Ra = []

    for i in range(mm):
        c = setting[i%len(setting)]
        if c == 'i':
            Ra.append(np.array([1-it for it in Rh[rn.randint(len(Rh))]]))
        elif c == 'r':
            Ra.append(gen_random(n))
        elif c == 'p':
            Ra.append(gen_honest_reviews(GT, 1, p)[0])
        else:
            Ra.append(np.zeros(n))

    return Ra

def gen_targeted_reviews(GT, mm, p, tar):
    return [np.array(list(r[:tar]) + [1-GT[tar]] + list(r[tar+1:])) for r in gen_honest_reviews(GT, mm, p)]

def hamm_dist(r1, r2): # Hamming distance between two reviews
    return np.count_nonzero(r1 != r2) # abs(r1-r2)[i] = 1 iff r1[i] != r2[i]

def estimate(R, ep, p, alpha):
    q = 1 - p
    m = len(R)
    n = len(R[0])

    t1 = time.time()
    tt = 0

    ##### STEP 1: REMOVE REVIEWS TOO FAR FROM THE REST #####
    neighbors = {i:1 for i in range(m)} # Number of reviews that each review is "close" to

    for i in range(m):
        ri = R[i]
        for j in range(i+1, m):
            rj = R[j]
            t3 = time.time()
            if hamm_dist(ri, rj) <= (1+ep)*2*p*q*n:
                tt += (time.time() - t3)
                neighbors[i] += 1
                neighbors[j] += 1

    R1 = [i for i in neighbors if neighbors[i] >= (1-alpha) * m] # Only take reviews that have enough neighbors

    if debugging:
        print("Step 1 took %.3f seconds." % (time.time() - t1))
        print("%.3f was spent on hamm_dist" % (tt))

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

    R2 = [R1[i] for i in range(len(R1)) if i not in marked2]
    if len(R2) == 0:
        raise ZeroDivisionError("The final set of reviews is empty. Increase your epsilon!")

    theta_hat = majority_vote([R[i] for i in R2])

    return theta_hat

def majority_vote(R):
    return np.int32(np.round(np.sum(np.array(R), axis=0) / len(R)))

def estimate_p(R):
    medians = sorted([np.median([hamm_dist(r, r1) for r1 in R]) for r in R])
    D_tilde = np.median(medians)# - (medians[-1]-medians[0])
    #D_tilde = sorted(medians)[0]
    return 0.5 + np.sqrt(1-(2*D_tilde/len(R[0])))/2#, 0.5 + np.sqrt(1-2*np.quantile(medians, 0.75)/len(R[0]))/2

debugging = False

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
    m = 200

    alpha = 0.34
    p = 0.75
    ep = 0.3

    GT = gen_random(n)
    Rh_init = np.array(list(GT) * alpha*m)


    Rh = gen_honest_reviews(GT, (1-alpha)*m, p, Rh_init)
    Ra = gen_targeted_reviews(GT, alpha*m, p, 0)

    R = Rh + Ra
    rn.shuffle(R)

    t1 = time.time()

    est = estimate(R, ep, p, alpha)

    score = est[0] == GT[0]#(1 - (hamm_dist(estimate(R, ep, p, alpha), GT) / n))*100

    if not debugging:
        print("___________")
        print("| n: %d" % n)
        print("| m: %d" % m)
        print("| p: %.3f" % p)
        print("| α: %.3f" % alpha)
        print("| ε: %.3f" % ep)
    print("SCORE: %.1f%%. Executed in %.3f seconds." % (score, time.time() - t1))

#main()
