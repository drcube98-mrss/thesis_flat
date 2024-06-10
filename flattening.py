import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def berry_curvature(H, N, nocc=1):
    kxs, dkx = np.linspace(-np.pi, np.pi, N, endpoint=False, retstep=True)
    kys, dky = np.linspace(-np.pi, np.pi, N, endpoint=False, retstep=True)
    ham = np.array([[H(kx, ky) for ky in kys] for kx in kxs])
    eigvals, eigvecs = la.eigh(ham)
    eigvecs = eigvecs[:, :, :, :nocc]
    P = eigvecs @ np.transpose(eigvecs.conj(), (0, 1, 3, 2))
    DP = np.array([
        (np.roll(P, -1, a) - np.roll(P, 1, a)) / (2 * dxa)
        for a, dxa in zip((0, 1), (dkx, dky))])
    eta = np.trace(DP[:, np.newaxis] @ DP[np.newaxis, :] @ P[np.newaxis, np.newaxis], axis1=-1, axis2=-2)
    bc = -(eta[0, 1] - eta[1, 0]).imag
    return bc

def iterate_flatten_curvature(H, N, iterations=10):
    A = np.zeros((N, N, 2))
    stdF = np.inf
    for i in range(iterations):
        B = np.fft.ifft2(np.fft.fftshift(F) / np.mean(F))
        N = F.shape[0]
        s = slice(0, N)
        neighbors = np.mgrid[s, s].T - np.array([1, 1]) * N//2
        denom = np.einsum('xyi, xyi -> xy', neighbors, N/(2 * np.pi) * np.sin((2 * np.pi) / N * neighbors))
        A_new = np.fft.fftshift(B)[:, :, np.newaxis] * neighbors / denom[:, :, np.newaxis]
        A_new = np.nan_to_num(A)
        A_new[0, :, 1] = A_new[:, 0, 0] = 0
        A_new = np.fft.fftshift(A_new, axes=(0, 1))
        A_new = (1j * np.fft.fft2(A_new, axes=(0, 1))).real

        def H_adj(kx, ky):
            k_ind = np.array(np.around([kx, ky] * (N / (2 * np.pi))), dtype=int)
            k_adj = np.array([kx, ky]) - A[k_ind[1] % N, k_ind[0] % N, :]
            return H(k_adj[0], k_adj[1])
        F = berry_curvature(H_adj, N)
        stdF = np.std(F)
        A += A_new
    return A, H
