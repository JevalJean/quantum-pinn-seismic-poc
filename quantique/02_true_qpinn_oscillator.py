import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# 1. Le Circuit Quantique (Identique, on garde notre machine)
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def quantum_neural_net(t, poids):
    qml.RY(t, wires=0)
    qml.RX(poids[0], wires=0)
    qml.RY(poids[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# 2. La Fonction de Coût "Physics-Informed" (La vraie magie)
def fonction_cout(poids):
    # L'IA fait sa prédiction sur tous les pas de temps
    u_pred = quantum_neural_net(T_train, poids)
    
    # --- CONTRAINTE 1 : La Physique (L'équation différentielle) ---
    # Calcul de l'accélération (dérivée seconde) par différences finies
    u_avant = u_pred[:-2]
    u_centre = u_pred[1:-1]
    u_apres = u_pred[2:]
    acceleration = (u_apres - 2 * u_centre + u_avant) / (dt ** 2)
    
    # Le résidu de l'équation : accélération + deplacement = 0
    residu_physique = acceleration + u_centre
    loss_physique = np.mean(residu_physique ** 2)
    
    # --- CONTRAINTE 2 : Conditions aux limites (Position initiale) ---
    # À t=0, le déplacement doit être 0 (u(0) = 0)
    loss_frontiere = (u_pred[0] - 0.0) ** 2
    
    # L'erreur totale est la somme des deux contraintes
    return loss_physique + loss_frontiere

# 3. Préparation du domaine temporel (Plus de Y_vrai !)
T_max = np.pi
N_points = 30
T_train = np.linspace(0, T_max, N_points, requires_grad=False)
dt = T_max / (N_points - 1) # Le pas de temps

# Poids initiaux au hasard
np.random.seed(42)
poids_actuels = np.random.randn(2, requires_grad=True)

# 4. L'Entraînement
optimiseur = qml.AdamOptimizer(stepsize=0.1)

print("Début de l'apprentissage auto-supervisé par la physique...")
for iteration in range(100):
    poids_actuels, cout = optimiseur.step_and_cost(fonction_cout, poids_actuels)
    
    if (iteration + 1) % 20 == 0:
        print(f"Itération {iteration + 1:3d} | Résidu Physique (Erreur) : {cout:.4f}")

# 5. Visualisation (On compare avec la théorie de la RDM)
U_pred_final = quantum_neural_net(T_train, poids_actuels)
# La solution analytique théorique de u'' + u = 0 avec u(0)=0 est un sinus pur
U_analytique = np.sin(T_train) 

plt.figure(figsize=(8, 5))
plt.plot(T_train, U_analytique, label="Solution Analytique Exacte", color='blue', linewidth=2)
plt.plot(T_train, U_pred_final, 'ro', label="Prédiction Q-PINN (Apprise sans données)", markersize=6)
plt.title("Le Q-PINN résout l'équation de l'oscillateur harmonique", fontsize=14)
plt.xlabel("Temps (s)", fontsize=12)
plt.ylabel("Déplacement u(t)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
