from algo1 import *

def sac_attack(GT, Rh, p, a):
    m = len(Rh) / (1-a)
    q = 1-p

    # WLOG, attack first item

    Ra = np.array([1-GT for i in range(int(np.ceil(a*m)))])
    qA = np.mean([(1-a)*(p-q)/a, 1-(p*(1-a)*(p-q))/a])

    for i in range(int(np.ceil(a*m*(1-qA)))): # Put in correct answers on sacrificial item
        Ra[i][0] = GT[0]

    return Ra

def query50(GT, R):
    splits = np.sum(R, axis=0) / np.shape(R)[0]
    queried = np.argmin(np.abs(splits-0.5))
    new_R = np.delete(R, [j for j in range(m) if R[j][queried] != GT[queried]], axis=0)
    return majority_vote(new_R)

alpha = 0.43
p = 0.7
m = 10000

GT = np.ones(20)
Rh_init = np.array(list(GT) * int((1-alpha)*m))
"""Rh = gen_honest_reviews(GT, (1 - alpha) * m, p)
print(np.sum(Rh, axis=0) / np.shape(Rh)[0])"""

run = True

if run:
    for i in range(100):
        Rh = gen_honest_reviews(GT, np.floor((1 - alpha) * m), p, Rh_init)
        Ra = sac_attack(GT, Rh, p, alpha)

        R = list(Rh) + list(Ra)
        rn.shuffle(R)

        splits = np.sum(R, axis=0) / np.shape(R)[0]
        diffs = np.abs(splits - 0.5)

        #print(i, diffs[1]-diffs[0])
        print(query50(GT,R)[:])
        if query50(GT,R)[1] != 0:
            print(i, np.abs(splits - 0.5))
            break


# TODO: why failure above lower bound???!?!?

