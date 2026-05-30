import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# 1. Le Circuit Quantique (Maintenant capable de traiter des vecteurs !)
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def quantum_neural_net(donnees_x, poids):
    # donnees_x est un vecteur entier. Le circuit va créer 20 états parallèles.
    qml.RY(donnees_x, wires=0)
    qml.RX(poids[0], wires=0)
    qml.RY(poids[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# 2. La Fonction de Coût (Simplifiée à l'extrême)
# On ne passe QUE les poids en argument, pour que l'optimiseur ne surveille qu'eux.
def fonction_cout(poids):
    # Plus aucune boucle for ! Le circuit quantique calcule les 20 prédictions d'un coup
    predictions = quantum_neural_net(X_train, poids)
    # L'erreur MSE est calculée vectoriellement
    return np.mean((predictions - Y_vrai) ** 2)

# 3. Préparation des données 
X_train = np.linspace(-np.pi, np.pi, 20, requires_grad=False)
Y_vrai = np.sin(X_train, requires_grad=False)

# Poids initiaux
poids_actuels = np.array([0.5, -0.2], requires_grad=True)

# 4. L'Entraînement
optimiseur = qml.AdamOptimizer(stepsize=0.1)

print("Début de l'entraînement vectorisé...")
for iteration in range(50):
    # L'optimiseur ne prend plus que la fonction de coût et les poids.
    poids_actuels, cout = optimiseur.step_and_cost(fonction_cout, poids_actuels)
    
    if (iteration + 1) % 10 == 0:
        print(f"Itération {iteration + 1:2d} | Erreur MSE : {cout:.4f}")

# 5. Visualisation
Y_pred = quantum_neural_net(X_train, poids_actuels)

plt.figure(figsize=(8, 5))
plt.plot(X_train, Y_vrai, label="Vraie dynamique (Sinus)", color='blue', linewidth=2)
plt.plot(X_train, Y_pred, 'ro', label="Prédictions Quantiques", markersize=8, alpha=0.8)
plt.title("Q-PINN : Apprentissage via Batching Quantique", fontsize=14)
plt.xlabel("Temps / Donnée d'entrée (x)", fontsize=12)
plt.ylabel("Déformation / Sortie (y)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
