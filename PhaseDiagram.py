import numpy as np
from scipy.linalg import eigh 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable

def main():
    z = 4
    n_max = 31

    epsilon = 1e-5

    phi_e, phi_o, theta = 0.001, 0.002, 0.0

    n = np.arange(0, n_max + 1)
    n_diag = np.diag(n)

    def hamiltonian_even(phi_e, phi_o, theta, mu_U0, zt_U0, U_inf_U0):
        a = np.diag(np.sqrt(np.arange(1, n_max + 1)), 1) 
        a_dag = a.T 
        
        identity_matrix = np.eye(n_max + 1)
        
        H_e = -zt_U0 * phi_o * (a + a_dag) + zt_U0 * phi_o * phi_e * identity_matrix \
            + 0.5 * (n_diag @ (n_diag - identity_matrix)) \
            - U_inf_U0 * theta * n_diag \
            + (U_inf_U0 / 4) * theta**2 * identity_matrix \
            - mu_U0 * n_diag
            
        return H_e

    def hamiltonian_odd(phi_e, phi_o, theta, mu_U0, zt_U0, U_inf_U0):
        a = np.diag(np.sqrt(np.arange(1, n_max + 1)), 1) 
        a_dag = a.T 
        
        identity_matrix = np.eye(n_max + 1)
        
        H_o = -zt_U0 * phi_e * (a + a_dag) + zt_U0 * phi_e * phi_o * identity_matrix \
            + 0.5 * (n_diag @ (n_diag - identity_matrix)) \
            + U_inf_U0 * theta * n_diag \
            + (U_inf_U0 / 4) * theta**2 * identity_matrix \
            - mu_U0 * n_diag
            
        return H_o

    def ground_state_expectation(hamiltonian):
        eigvals, eigvecs = eigh(hamiltonian) 
        ground_state = eigvecs[:, 0] 
        n_expect = np.vdot(ground_state, np.dot(np.diag(n), ground_state))  
        a_expect = np.vdot(ground_state, np.dot(np.diag(np.sqrt(np.arange(1, n_max + 1)), 1), ground_state))
        energy = np.vdot(ground_state, np.dot(hamiltonian, ground_state))  
        
        return a_expect, n_expect, energy

    def fixed_point_iteration(mu_U0, zt_U0, U_inf_U0, phi_e, phi_o, theta, max_iters=300):
        for i in range(max_iters):
            H_e = hamiltonian_even(phi_e, phi_o, theta, mu_U0, zt_U0, U_inf_U0)
            H_o = hamiltonian_odd(phi_e, phi_o, theta, mu_U0, zt_U0, U_inf_U0)

            a_e, n_e, E_e = ground_state_expectation(H_e)
            a_o, n_o, E_o = ground_state_expectation(H_o)

            energy = 0.5 * (E_o + E_e)
            phi_e_new, phi_o_new = a_e, a_o
            theta_new = n_e - n_o

            delta_phi_e = np.abs(phi_e_new - phi_e)
            delta_phi_o = np.abs(phi_o_new - phi_o)
            delta_theta = np.abs(theta_new - theta)

            phi_e, phi_o, theta = phi_e_new, phi_o_new, theta_new
            
            if max(delta_phi_e, delta_phi_o, delta_theta) < epsilon:
                break
        density = (n_e + n_o) / 2


        return phi_e, phi_o, theta, density, energy



    def plot_parameters_as_functions_of_mu(U_inf_U0, zt_U0, min_range, max_range, maxsteps):
        # plots en función de mu
        mu_U0_values = np.linspace(min_range, max_range, maxsteps)

        varphi_values = []
        theta_abs_values = []
        density_values = []

        min_energy = np.inf
        best_params = None

        i=1
        for mu_U0 in mu_U0_values:
            print("step", i, "of", maxsteps)
            i+=1
            # Initial Guesses to compare minimal energy
            for N_theta in range(24):
                for initial_guess in [(0, 0, N_theta), (0.001, 0.002, N_theta), (0.1, 0.2, N_theta)]:
                    phi_e, phi_o, theta = initial_guess
                    phi_e, phi_o, theta, density, energy = fixed_point_iteration(mu_U0, zt_U0, U_inf_U0, phi_e, phi_o, theta)
                    if energy < min_energy:
                        min_energy = energy
                        best_params = (phi_e, phi_o, theta, density, energy)
            #print(f"minimum energy at initial guess: {initial_guess}")

            varphi_values.append(np.sqrt(np.abs(best_params[0] * best_params[1])))
            theta_abs_values.append(np.abs(best_params[2]))
            density_values.append(best_params[3])

        varphi_values = np.array(varphi_values)
        theta_abs_values = np.array(theta_abs_values)
        density_values = np.array(density_values)

        plt.figure(figsize=(8, 5))
        plt.plot(mu_U0_values, varphi_values, label='varphi', color='blue', marker='.', linestyle='')
        plt.xlabel('mu/U_0', fontsize=14)
        plt.ylabel('varphi', fontsize=14)
        plt.title('varphi (mu)', fontsize=16)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(mu_U0_values, theta_abs_values, label='|theta|', color='red', marker='.', linestyle='')
        plt.xlabel('mu/U_0', fontsize=14)
        plt.ylabel('|theta|', fontsize=14)
        plt.title('|theta| (mu)', fontsize=16)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(mu_U0_values, density_values, label='Density (rho)', color='green', marker='.', linestyle='')
        plt.xlabel('mu/U_0', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Density (mu)', fontsize=16)
        plt.grid(True)
        plt.show()

    #plot_parameters_as_functions_of_mu(0.87, 0.66, -.5, -.3)
    #plot_parameters_as_functions_of_mu(0.45, 0.35, -.2, -.05)
    #plot_parameters_as_functions_of_mu(0.555, 0.32, .25, .35)

    def plot_phase_diagram(zt_range, mu_range, U_inf_U0 = 0.6, resolution=80):
        # Initialize arrays for zt/U_0, mu/U_0, and varphi
        zt_values = np.linspace(*zt_range, resolution)
        mu_values = np.linspace(*mu_range, resolution)
        varphi_matrix = np.zeros((resolution, resolution))
        theta_matrix = np.zeros((resolution, resolution))

        # Define initial conditions to check
        N_theta = 24
        initial_conditions = [(0, 0, n) for n in range(N_theta)] + \
                            [(0.001, 0.002, n) for n in range(N_theta)] + \
                            [(0.1, 0.2, n) for n in range(N_theta)]


        step = 0
        for i, zt_U0 in enumerate(zt_values):
            for j, mu_U0 in enumerate(mu_values):
                print('step',step + 1, '/', resolution**2)
                step += 1
                min_energy = np.inf
                best_varphi = 0
                best_theta = 0

                for initial in initial_conditions:
                    phi_e, phi_o, theta = initial
                    phi_e, phi_o, theta, density, energy = fixed_point_iteration(mu_U0, zt_U0, U_inf_U0, phi_e, phi_o, theta)

                    if energy < min_energy:
                        min_energy = energy
                        best_varphi = np.sqrt(np.abs(phi_e * phi_o))
                        best_theta = abs(theta)

                varphi_matrix[j, i] = best_varphi
                theta_matrix[j, i] = best_theta


        fig, ax = plt.subplots(figsize=(8, 6))
        norm = Normalize(vmin=np.min(varphi_matrix), vmax=1.5)
        cbar = ScalarMappable(norm=norm, cmap='viridis')

        c = ax.pcolormesh(zt_values, mu_values, varphi_matrix, shading='auto', cmap='viridis', norm=norm)
        ax.set_xlabel('zt/U_0', fontsize=14)
        ax.set_ylabel('mu/U_0', fontsize=14)
        ax.set_title('varphi as a function of zt/U_0 and mu/U_0 with U_infty/U_0=0.6', fontsize=16)
        fig.colorbar(c, ax=ax, label='varphi')
        plt.show()

        #theta
        fig, ax = plt.subplots(figsize=(8, 6))
        norm = Normalize(vmin=np.min(theta_matrix), vmax=2)
        cbar = ScalarMappable(norm=norm, cmap='viridis')

        c = ax.pcolormesh(zt_values, mu_values, theta_matrix, shading='auto', cmap='viridis', norm=norm)
        ax.set_xlabel('zt/U_0', fontsize=14)
        ax.set_ylabel('mu/U_0', fontsize=14)
        ax.set_title('theta as a function of zt/U_0 and mu/U_0 with U_infty/U_0=0.6', fontsize=16)
        fig.colorbar(c, ax=ax, label='|theta|')
        plt.show()

    # Example usage
    #plot_phase_diagram(zt_range=(0.1, 0.3), mu_range=(-0.3, 0.3))
    #plot_phase_diagram(zt_range=(0.2, 0.6), mu_range=(-0.4, 0.4))

    def plot_full_phase_diagram(U_inf_U0, zt_range, mu_range, resolution):
        zt_values = np.linspace(*zt_range, resolution)
        mu_values = np.linspace(*mu_range, resolution)
        phase_matrix = np.zeros((resolution, resolution))

        N_theta = 20
        initial_conditions = [(0, 0, n) for n in range(N_theta)] + \
                            [(0.0001, 0.0002, n) for n in range(N_theta)] + \
                            [(0.1, 0.2, n) for n in range(N_theta)]

        step = 0
        for i, zt_U0 in enumerate(zt_values):
            for j, mu_U0 in enumerate(mu_values):
                print('step', step + 1, '/', resolution**2)
                step += 1
                min_energy = np.inf
                best_varphi = 0
                best_theta = 0

                for initial in initial_conditions:
                    phi_e, phi_o, theta = initial
                    phi_e, phi_o, theta, density, energy = fixed_point_iteration(mu_U0, zt_U0, U_inf_U0, phi_e, phi_o, theta)

                    if energy < min_energy:
                        min_energy = energy
                        best_varphi = np.sqrt(np.abs(phi_e * phi_o))
                        best_theta = abs(theta)
                #print(zt_U0, mu_U0, best_varphi, best_theta)
                if best_theta > 0.01 and best_varphi < 0.01:
                    phase_matrix[j, i] = 1  # CDW
                elif best_theta < 0.01 and best_varphi > 0.01:
                    phase_matrix[j, i] = 4  # SF
                elif best_theta > 0.01 and best_varphi > 0.01:
                    phase_matrix[j, i] = 3  # SS
                elif best_theta < 0.01 and best_varphi < 0.01:
                    phase_matrix[j, i] = 2  # MI

        cmap = ListedColormap([ '#999AC6', '#B8BACF', '#d2d5dd', '#E8EBE4'])
        cmap = 'gist_heat'
        norm = Normalize(vmin=0, vmax=5)

        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.pcolormesh(zt_values, mu_values, phase_matrix, shading='auto', cmap=cmap, norm=norm)
        ax.set_xlabel('zt/U_0', fontsize=14)
        ax.set_ylabel('mu/U_0', fontsize=14)
        ax.set_title('U_LR/U_0=0.6', fontsize=16)
        cbar = fig.colorbar(c, ax=ax, ticks=[1.5, 2.5, 3.5, 4.5])
        cbar.ax.set_yticklabels(['CDW', 'MI', 'SS', 'SF'])
        plt.show()


    #plot_full_phase_diagram(0.3, zt_range=(0.1, 0.3), mu_range=(-0.2, 0.6))
    #plot_full_phase_diagram(0.6, zt_range=(0.2, 0.6), mu_range=(-0.3, 0.7))



    print("Phase Diagram for Bose-Hubbard model exended with cavity by mean-field approach\n")
    while True:
        operation = input('Choose operation\n [1] Find mean-field parameters for one point \n [2] Mean-field parameters as funtions of mu \n [3] Phase Diagram \n operation: ')
        if operation == '1':
            print('Input interaction terms:\n')
            mu_U0 = float(input('value of mu/U: '))
            zt_U0 = float(input('value of zt/U: '))
            U_inf_U0 = float(input('value of U_cav/U: '))
            min_energy = np.inf
            best_params = None                
            for N_theta in range(24):
                for initial_guess in [(0, 0, N_theta), (0.001, 0.002, N_theta), (0.1, 0.2, N_theta)]:
                    phi_e, phi_o, theta = initial_guess
                    phi_e, phi_o, theta, density, energy = fixed_point_iteration(mu_U0, zt_U0, U_inf_U0, phi_e, phi_o, theta)
                    if energy < min_energy:
                        min_energy = energy
                        best_params = (phi_e, phi_o, theta, density, energy)
            print('(phi_e, phi_o, theta, density, energy)=',best_params)
            
            break

        elif operation == '2':
            print('Input interaction terms and range:\n')

            zt_U0 = float(input('value of zt/U: '))
            U_inf_U0 = float(input('value of U_cav/U: '))
            min_range = float(input('min mu: '))
            max_range = float(input('max mu: '))
            resolution = int(input('resolution: '))
            plot_parameters_as_functions_of_mu(U_inf_U0, zt_U0, min_range, max_range, resolution)
            break

        elif operation == '3':
            print('Input terms and ranges:\n')

            min_zt = float(input('min zt/U: '))
            max_zt = float(input('max zt/U: '))
            min_mu = float(input('min mu: '))
            max_mu = float(input('max mu: '))
            zt_range=(min_zt, max_zt)
            mu_range=(min_mu, max_mu)
            resolution = int(input('resolution (suggested <80): '))
            plot_full_phase_diagram(U_inf_U0, zt_range, mu_range, resolution)
            break



if __name__ == "__main__":
    main()