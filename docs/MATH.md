# CKKS Mathematical Reference

## Table of Contents
1. [Ring Theory Basics](#ring-theory-basics)
2. [Encoding and Decoding](#encoding-and-decoding)
3. [Encryption Scheme](#encryption-scheme)
4. [Homomorphic Operations](#homomorphic-operations)
5. [Relinearization](#relinearization)
6. [Rescaling](#rescaling)
7. [Bootstrapping](#bootstrapping)

---

## Ring Theory Basics

### Polynomial Ring

The CKKS scheme operates on the ring:

$$R = \mathbb{Z}_q[X]/(X^N + 1)$$

where:
- $N$ is a power of 2 (polynomial degree)
- $q$ is the ciphertext modulus (product of primes)
- $X^N \equiv -1$ (negacyclic convolution)

### Roots of Unity

The polynomial $X^N + 1$ has $2N$ roots called **N-th roots of unity**:

$$\zeta_{2N}^k = e^{i\pi k/N}, \quad k = 0, 1, \ldots, 2N-1$$

where $\zeta_{2N} = e^{i\pi/N}$ is the primitive $2N$-th root.

### Discrete Fourier Transform

For vectors $\mathbf{z} = (z_1, \ldots, z_{N/2}) \in \mathbb{C}^{N/2}$, the inverse DFT gives:

$$\text{DFT}^{-1}(\mathbf{z})_j = \sum_{k=0}^{N/2-1} z_k \cdot \zeta_{2N}^{jk}, \quad j = 0, \ldots, N-1$$

The forward DFT recovers the values.

---

## Encoding and Decoding

### Encoding Process

1. **Input**: Complex vector $\mathbf{z} = (z_1, \ldots, z_{N/2})$

2. **Conjugate Symmetry**: Create full vector with conjugate pairs:
   $$\tilde{z}_j = \begin{cases} z_j & j < N/2 \\ \overline{z}_{N-j} & j \geq N/2 \end{cases}$$

3. **Scaling**: Multiply by scaling factor $\Delta$:
   $$\tilde{z}'_j = \Delta \cdot \tilde{z}_j$$

4. **Round to Integers**: $\mathbf{m} = \lfloor \tilde{z}' \rceil$

5. **Inverse DFT**: Apply inverse DFT to get polynomial coefficients

6. **Output**: Polynomial $m(X) = \sum_{j=0}^{N-1} m_j X^j$

### Decoding Process

1. **Input**: Polynomial coefficients $\mathbf{m}$

2. **Forward DFT**: Recover complex values

3. **Divide by $\Delta$**: $\tilde{z}_j = m_j / \Delta$

4. **Extract Slots**: Take first $N/2$ values

---

## Encryption Scheme

### Secret Key Generation

Sample polynomial $\mathbf{s}$ from ternary distribution:
$$s_i \in \{-1, 0, 1\}$$
with hamming weight $H \approx N/4$.

### Public Key Generation

1. Sample $\mathbf{a} \leftarrow \mathcal{U}(\mathbb{Z}_q^N)$
2. Sample error $\mathbf{e} \leftarrow \chi(\mathbb{Z}^N)$ (discrete Gaussian)
3. Compute:
   $$\mathbf{b} = -\mathbf{a} \cdot \mathbf{s} + \mathbf{e} \pmod{q}$$

Public key: $(\mathbf{a}, \mathbf{b})$

### Encryption

For message polynomial $\mathbf{m}$:

1. Sample $\mathbf{u} \leftarrow \mathcal{U}(\{-1, 0, 1\}^N)$
2. Sample errors $\mathbf{e}_0, \mathbf{e}_1 \leftarrow \chi$
3. Compute:
   $$c_0 = \mathbf{b} \cdot \mathbf{u} + \mathbf{m} + \mathbf{e}_0 \pmod{q}$$
   $$c_1 = \mathbf{a} \cdot \mathbf{u} + \mathbf{e}_1 \pmod{q}$$

Ciphertext: $(\mathbf{c}_0, \mathbf{c}_1)$

### Decryption

$$\mathbf{m}' = \mathbf{c}_0 + \mathbf{c}_1 \cdot \mathbf{s} \pmod{q}$$

---

## Homomorphic Operations

### Addition

For ciphertexts $ct = (\mathbf{c}_0, \mathbf{c}_1)$ and $ct' = (\mathbf{c}'_0, \mathbf{c}'_1)$:

$$ct_{add} = (\mathbf{c}_0 + \mathbf{c}'_0, \mathbf{c}_1 + \mathbf{c}'_1) \pmod{q}$$

Complexity: $O(N)$

### Multiplication

For fresh ciphertexts:
$$ct_{mult} = (\mathbf{c}_0 \cdot \mathbf{c}'_0, \mathbf{c}_0 \cdot \mathbf{c}'_1 + \mathbf{c}_1 \cdot \mathbf{c}'_0, \mathbf{c}_1 \cdot \mathbf{c}'_1)$$

This produces a **degree-2** ciphertext requiring relinearization.

Complexity: $O(N \log N)$

### Rotation (Slot Permutation)

Rotation corresponds to automorphism:
$$X^k \mapsto X^{k \cdot r} \pmod{X^N + 1}$$

This enables efficient slot permutations for SIMD operations.

---

## Relinearization

After multiplication, ciphertext has form:
$$ct = (d_0, d_1, d_2)$$
with decryption:
$$m = d_0 + d_1 \cdot s + d_2 \cdot s^2$$

### Key Switching

Express $d_2$ in gadget basis:
$$d_2 = \sum_{i=0}^{L-1} \tau_i(d_2) \cdot \mathbf{g}^i$$

where $\mathbf{g} = (1, B, B^2, \ldots)$ is the gadget vector.

### Relinearization Key

Precomputed as:
$$rlk_i = (\mathbf{b}_i, \mathbf{a}_i) = (-\mathbf{a}_i \cdot \mathbf{s} + \mathbf{e}_i + B^i \cdot \mathbf{s}^2, \mathbf{a}_i)$$

### Result

After relinearization:
$$ct_{relin} = (d_0 + \sum_i \tau_i(d_2) \cdot \mathbf{b}_i, \sum_i \tau_i(d_2) \cdot \mathbf{a}_i)$$

Back to degree-1, can continue operations.

---

## Rescaling

After multiplication, scaling factor doubles:
$$\Delta_{new} = \Delta_{old}^2$$

### Rescale Operation

$$ct_{new} = \left\lfloor \frac{\mathbf{c}_0}{\Delta} \right\rceil, \left\lfloor \frac{\mathbf{c}_1}{\Delta} \right\rceil$$

This:
1. Reduces scaling factor back to $\Delta$
2. Removes some noise
3. Switches to lower modulus level

### Modulus Switching

Switch from modulus $q$ to $q'$:
$$ct' = \left\lfloor \frac{q'}{q} \cdot \mathbf{c} \right\rceil \pmod{q'}$$

This reduces noise but loses levels for multiplication.

---

## Bootstrapping

Bootstrapping "refreshes" a ciphertext by homomorphically computing its decryption.

### Why Bootstrapping?

Each multiplication adds noise. After $L$ levels, ciphertext becomes too noisy.

Bootstrapping resets to fresh state, enabling unlimited depth.

### Bootstrapping Algorithm

1. **Modulus Switching**: Switch to smallest modulus $q_0$

2. **Encode for Slots**: Replicate single decryption to all slots using rotations

3. **Modular Reduction**: Approximate $x \mod q_0$ using sine polynomial:
   $$\sin(2\pi x / q_0) \approx 0 \text{ at multiples of } q_0$$

4. **Re-encrypt**: Encrypt the approximated plaintext with fresh noise

### Sine Approximation

The sine function has zeros at integers:
$$\sin(2\pi k) = 0, \quad k \in \mathbb{Z}$$

Use this to detect and correct modular reduction errors.

---

## Error Analysis

### Noise Distribution

After operations, noise $\mathbf{e}$ grows:

| Operation | Noise Growth |
|-----------|--------------|
| Addition | $\|\mathbf{e}\| + \|\mathbf{e}'\|$ |
| Multiplication | $B \cdot (\|\mathbf{e}\| + \|\mathbf{e}'\|)$ |

where $B$ depends on ciphertext magnitude.

### Precision

Approximate error standard deviation:
$$\sigma_{final} = \sqrt{\sigma_0^2 + L \cdot \sigma_{mult}^2}$$

where $L$ is multiplicative depth.

### Maximum Precision Bits

$$\text{Precision} \approx \log_2\left(\frac{\Delta}{\sigma_{final}}\right)$$

Typical values: 20-40 bits depending on depth.

---

## Security Proof Sketch

### RLWE Hardness

The CKKS scheme's security reduces to the **Ring Learning With Errors (RLWE)** problem:

**Decision RLWE**: Distinguish $(\mathbf{a}, \mathbf{a} \cdot \mathbf{s} + \mathbf{e})$ from uniform.

**Search RLWE**: Recover $\mathbf{s}$ from samples.

### Reduction

1. Public key is RLWE sample with secret $\mathbf{s}$
2. Ciphertext is RLWE sample with same secret
3. Operations preserve RLWE structure

### Concrete Security

For $N = 2^{15}$ and $\log q \approx 300$:
- Classical attack: exponential in $N$
- Quantum attack: exponential in $\sqrt{N}$

Current estimates: ~128-256 bits of security.

---

## References

1. Cheon, J. H., et al. (2017). "Homomorphic Encryption for Arithmetic of Approximate Numbers." ASIACRYPT 2017.

2. Halevi, S., & Shoup, V. (2018). "Design and Implementation of HEAAN." IACR Cryptology ePrint Archive.

3. Cheon, J. H., et al. (2019). "Bootstrapping for HEAAN." EUROCRYPT 2018.

4. Homomorphic Encryption Standard. homomorphicencryption.org
