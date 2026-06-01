import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# 1. Le Simulateur NISQ (Gestion du bruit avec default.mixed)
dev = qml.device("default.mixed", wires=1)

# On définit un niveau de bruit (Ici 5% de perte d'information)
prob_bruit = 0.05

@qml.qnode(dev)
def quantum_neural_net(t, poids):
    # Encodage et ansatz standard
    qml.RY(t, wires=0)
    qml.RX(poids[0], wires=0)
    qml.RY(poids[1], wires=0)
    
    # --- INJECTION DU BRUIT ---
    # Le canal de dépolarisation simule l'interaction avec l'environnement
    qml.DepolarizingChannel(prob_bruit, wires=0)
    
    return qml.expval(qml.PauliZ(0))

# 2. La Fonction de Coût "Physics-Informed" (Inchangée)
def fonction_cout(poids):
    u_pred = quantum_neural_net(T_train, poids)
    
    u_avant = u_pred[:-2]
    u_centre = u_pred[1:-1]
    u_apres = u_pred[2:]
    acceleration = (u_apres - 2 * u_centre + u_avant) / (dt ** 2)
    
    residu_physique = acceleration + u_centre
    loss_physique = np.mean(residu_physique ** 2)
    
    loss_frontiere = (u_pred[0] - 0.0) ** 2
    
    return loss_physique + loss_frontiere

# 3. Préparation du domaine
T_max = np.pi
N_points = 30
T_train = np.linspace(0, T_max, N_points, requires_grad=False)
dt = T_max / (N_points - 1)

np.random.seed(42)
poids_actuels = np.random.randn(2, requires_grad=True)

# 4. L'Entraînement sous contrainte de bruit
optimiseur = qml.AdamOptimizer(stepsize=0.1)

print(f"Début de l'apprentissage avec {prob_bruit*100}% de bruit quantique...")
for iteration in range(100):
    poids_actuels, cout = optimiseur.step_and_cost(fonction_cout, poids_actuels)
    
    if (iteration + 1) % 20 == 0:
        print(f"Itération {iteration + 1:3d} | Résidu Physique : {cout:.4f}")

# 5. Visualisation
U_pred_bruite = quantum_neural_net(T_train, poids_actuels)
U_analytique = np.sin(T_train) 

plt.figure(figsize=(8, 5))
plt.plot(T_train, U_analytique, label="Solution Exacte", color='blue', linewidth=2)
plt.plot(T_train, U_pred_bruite, 'rx', label="Q-PINN (Environnement Bruitée)", markersize=6)
plt.title(f"Résilience du Q-PINN face au bruit quantique ({prob_bruit*100}%)", fontsize=14)
plt.xlabel("Temps (s)", fontsize=12)
plt.ylabel("Déplacement u(t)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
