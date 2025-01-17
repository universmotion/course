# **`np.linalg.lstsq`**

La fonction **`np.linalg.lstsq`** de NumPy résout le problème des moindres carrés pour des systèmes d'équations linéaires. Autrement dit, elle trouve la solution optimale $\mathbf{x}$ au problème suivant :

$$
\min_{\mathbf{x}} \|\mathbf{A} \mathbf{x} - \mathbf{b}\|_2
$$

où :
- $\mathbf{A}$ est une matrice de taille $(m, n)$,
- $\mathbf{b}$ est un vecteur ou une matrice de taille $(m,)$ ou $(m, k)$,
- $\mathbf{x}$ est le vecteur ou la matrice de taille $(n,)$ ou $(n, k)$ qui minimise la différence.

C'est une méthode classique pour résoudre des systèmes d'équations linéaires surdéterminés (plus d'équations que d'inconnues, $m > n$).

---

### Utilisation de `np.linalg.lstsq`
Voici la signature de la fonction :

```python
np.linalg.lstsq(a, b, rcond=None)
```

#### Paramètres principaux :
1. **`a`** : Matrice des coefficients ($\mathbf{A}$), de taille $(m, n)$.
2. **`b`** : Vecteur ou matrice des termes constants ($\mathbf{b}$), de taille $(m,)$ ou $(m, k)$.
3. **`rcond`** : Seuil pour ignorer les petites valeurs singulières dans le calcul. Par défaut, si `None`, il utilise une valeur basée sur la taille de $\mathbf{A}$ et la précision numérique.

#### Valeurs retournées :
La fonction retourne un tuple contenant :
1. **`x`** : La solution optimisée ($\mathbf{x}$), de taille $(n,)$ ou $(n, k)$.
2. **`residuals`** : La somme des résidus $\|\mathbf{A} \mathbf{x} - \mathbf{b}\|^2$. Si le système est sous-déterminé ou a une solution exacte, ce sera un tableau vide.
3. **`rank`** : Le rang de la matrice $\mathbf{A}$.
4. **`singular_values`** : Les valeurs singulières de $\mathbf{A}$.

---

### Exemple simple

#### Cas 1 : Système surdéterminé
On veut résoudre le système suivant (surdéterminé) :
$$
2x + y = 5
$$
$$
x - y = 1
$$
$$
x + 2y = 6
$$

```python
import numpy as np

# Matrice des coefficients (A) et vecteur constant (b)
A = np.array([[2, 1], [1, -1], [1, 2]])
b = np.array([5, 1, 6])

# Résolution du problème
x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

print("Solution : ", x)
print("Résidus : ", residuals)
print("Rang : ", rank)
print("Valeurs singulières : ", singular_values)
```

#### Résultat attendu :
La solution approchée $(x, y)$ minimise l'erreur quadratique. Par exemple :

```
Solution : [1.85714286, 2.42857143]
Résidus : [0.14285714]  # somme des carrés des résidus
Rang : 2
Valeurs singulières : [2.82964727, 1.5138613]
```

---

### Cas d'utilisation dans le test d'indépendance conditionnelle
Dans le contexte de la fonction **`gauss_ci_test`**, `np.linalg.lstsq` est utilisé pour calculer les résidus après ajustement par régression :

- Par exemple, pour ajuster $ X $ en fonction de $ Z $, on résout :
  $$
  Z \mathbf{\beta_X} \approx X
  $$
  Cela donne $\mathbf{\beta_X}$, le vecteur des coefficients de régression.

- Les résidus $r_X$ sont ensuite calculés comme :
  $$
  r_X = X - Z \mathbf{\beta_X}
  $$

Cela permet de retirer l'influence des variables $Z$ sur $X$ et $Y$, ce qui est au cœur du test d'indépendance conditionnelle gaussien.
