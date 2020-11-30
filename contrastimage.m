example_MULTEM_HRTEM_perfect_MoS2_lattice;
example_MULTEM_HRTEM_defective_MoS2_lattice;
difference = perfect - defective;
imagesc(difference);
colormap gray;
figure(1)