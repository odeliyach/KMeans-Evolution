import sys
import numpy as np
import symnmfmodule


def init_H(W, k):
    """
    Initializes the H matrix for the SymNMF algorithm.
    H is of shape (N, k) with values in range [0, 2*sqrt(m/k)],
    where m is the mean of W and N is the number of data points.
    """
    np.random.seed(1234)
    m = np.mean(W)
    N = W.shape[0]
    initH = np.random.uniform(0, 2 * np.sqrt(m / k), size=(N, k))
    return initH


def main():
    """Main program: Handles input arguments and performs one of the spectral clustering goals using the symnmfmodule."""
    try:
        if len(sys.argv) != 4:
            raise ValueError
        k = int(sys.argv[1])
        goal = sys.argv[2]
        filename = sys.argv[3]
        if goal not in ["sym", "ddg", "norm", "symnmf"]:
            raise ValueError
        data = np.loadtxt(filename, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.size == 0:
            raise ValueError
        N, d = data.shape
        if k < 0 or k >= N:
            raise ValueError
        if goal == "symnmf":
            W = np.array(symnmfmodule.norm(data.tolist()))
            H = init_H(W, k)
            resultH = symnmfmodule.symnmf(H.tolist(), W.tolist(), 300, 1e-4)
            for row in resultH:
                print(",".join([f"{val:.4f}" for val in row]))
        elif goal == "sym":
            A = symnmfmodule.sym(data.tolist())
            for row in A:
                print(",".join([f"{val:.4f}" for val in row]))
        elif goal == "ddg":
            D = symnmfmodule.ddg(data.tolist())
            for row in D:
                print(",".join([f"{val:.4f}" for val in row]))
        elif goal == "norm":
            W = symnmfmodule.norm(data.tolist())
            for row in W:
                print(",".join([f"{val:.4f}" for val in row]))
        else:
            raise ValueError
    except:
        print("An Error Has Occurred")
        sys.exit(0) 

if __name__ == "__main__":
    main()
