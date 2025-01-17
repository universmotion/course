# Détail de la fonction **gauss_ci_test**

### 1. **Problème général des moindres carrés**
L'objectif est de résoudre :

$$
\min_{\mathbf{\beta}} \|\mathbf{A} \mathbf{\beta} - \mathbf{b}\|_2
$$

où :
- $\mathbf{A}$ est la matrice des prédicteurs (ici $Z$, les variables conditionnantes).
- $\mathbf{b}$ est le vecteur des valeurs observées (par exemple $X$ ou $Y$).
- $\mathbf{\beta}$ est le vecteur des coefficients de régression que l'on cherche à estimer.

Le résultat donne la meilleure approximation de $\mathbf{b}$ en termes de combinaison linéaire des colonnes de $\mathbf{A}$.

---

### 2. **Étapes de calcul de `np.linalg.lstsq`**

#### Étape 1 : Décomposition en valeurs singulières (SVD)
La fonction commence par calculer la **décomposition en valeurs singulières (SVD)** de $\mathbf{A}$ :
$$
\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$
où :
- $\mathbf{U}$ : matrice orthogonale ($m \times m$).
- $\mathbf{\Sigma}$ : matrice diagonale contenant les valeurs singulières ($m \times n$).
- $\mathbf{V}$ : matrice orthogonale ($n \times n$).

#### Étape 2 : Calcul de la solution
La solution des moindres carrés est donnée par :
$$
\mathbf{\beta} = \mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^T \mathbf{b}
$$
où :
- $\mathbf{\Sigma}^+$ est la pseudo-inverse de $\mathbf{\Sigma}$, calculée en inversant les valeurs singulières (les petites valeurs sont ignorées selon le seuil `rcond`).

#### Étape 3 : Résidus
Les résidus (différence entre les valeurs prédites et observées) sont calculés comme :
$$
\text{Résidus} = \|\mathbf{A} \mathbf{\beta} - \mathbf{b}\|^2
$$

#### Étape 4 : Rang et valeurs singulières
- Le **rang** est déterminé par le nombre de valeurs singulières significatives.
- Les **valeurs singulières** aident à diagnostiquer des problèmes comme une matrice mal conditionnée.

---

### 3. **Application dans le test d'indépendance conditionnelle**

#### Problème à résoudre
Lors du test d'indépendance conditionnelle, nous avons deux variables d'intérêt ($X$ et $Y$) et un ensemble de variables conditionnantes ($Z$). Le but est de vérifier si $X$ et $Y$ sont conditionnellement indépendants donné $Z$.

Pour cela, nous régressons $X$ et $Y$ sur $Z$, puis calculons leurs résidus, c'est-à-dire les parties de $X$ et $Y$ qui ne peuvent pas être expliquées par $Z$. Ces résidus sont ensuite utilisés pour calculer la corrélation partielle.

---

#### Étape 1 : Régression de $X$ sur $Z$
On résout :
$$
Z \mathbf{\beta_X} \approx X
$$

- $\mathbf{A} = Z$, la matrice des variables conditionnantes.
- $\mathbf{b} = X$, le vecteur de la variable cible.

Cela donne $\mathbf{\beta_X}$, les coefficients de régression, et on calcule les résidus :
$$
r_X = X - Z \mathbf{\beta_X}
$$

---

#### Étape 2 : Régression de $Y$ sur $Z$
De la même manière, on résout :
$$
Z \mathbf{\beta_Y} \approx Y
$$
et on obtient les résidus :
$$
r_Y = Y - Z \mathbf{\beta_Y}
$$

---

#### Étape 3 : Corrélation entre $r_X$ et $r_Y$
On calcule ensuite la corrélation entre les résidus $r_X$ et $r_Y$, ce qui correspond à la corrélation partielle entre $X$ et $Y$ conditionnellement à $Z$.

La corrélation partielle $r$ est utilisée pour calculer une statistique de test basée sur une distribution $t$-Student.

---

### Exemple de calcul détaillé

#### Données simulées
```python
import numpy as np

# Simulons des données
np.random.seed(42)
n = 100
Z = np.random.randn(n, 2)  # 2 variables conditionnantes
X = 2 * Z[:, 0] + np.random.randn(n)  # Dépend de Z avec du bruit
Y = -3 * Z[:, 1] + np.random.randn(n)  # Dépend de Z avec du bruit
```

#### Régression de $X$ sur $Z$
```python
# Résolution par moindres carrés
beta_X, _, _, _ = np.linalg.lstsq(Z, X, rcond=None)

# Calcul des résidus
resid_X = X - Z @ beta_X
```

#### Régression de $Y$ sur $Z$
```python
# Résolution par moindres carrés
beta_Y, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)

# Calcul des résidus
resid_Y = Y - Z @ beta_Y
```

#### Corrélation entre les résidus
```python
# Corrélation partielle
r = np.corrcoef(resid_X, resid_Y)[0, 1]
print(f"Corrélation partielle : {r}")
```

#### Statistique de test
On calcule la statistique $t$ et la p-valeur :
```python
from scipy.stats import t

# Degrés de liberté
dof = n - Z.shape[1] - 2

# Statistique t
stat = np.sqrt(dof) * r / np.sqrt(1 - r**2)

# P-valeur
p_value = 2 * t.sf(np.abs(stat), dof)
print(f"P-valeur : {p_value}")
print(f"Statistique t : {stat}")
```

---

### Conclusion
Les étapes clés consistent à :
1. Régresser $X$ et $Y$ sur $Z$ pour obtenir les résidus.
2. Calculer la corrélation entre les résidus ($r$).
3. Tester la significativité de cette corrélation via une statistique $t$.
