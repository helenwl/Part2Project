% output_multislice = il_MULTEM(system_conf, input_multislice) perform TEM simulation
% 
% High resolution transmission electron microscopy (HRTEM) simulation
% 
% All parameters of the input_multislice structure are explained in multem_default_values()
% 
% Copyright 2017 Ivan Lobato <Ivanlh20@gmail.com>
count=0
for i= 0:9
%clear mos2;
clc;
%%%%%%%%%%%%%%%%%% Load multem default parameter %%%%%%%%$$%%%%%%%%%
input_multislice = multem_default_values();          % Load default values;

%%%%%%%%%%%%%%%%%%%%% Set system configuration %%%%%%%%%%%%%%%%%%%%%
system_conf.precision = 1;                           % eP_Float = 1, eP_double = 2
system_conf.device = 2;                              % eD_CPU = 1, eD_GPU = 2
system_conf.cpu_nthread = 1; 
system_conf.gpu_device = 0;

%%%%%%%%%%%%%%%%%%%% Set simulation experiment %%%%%%%%%%%%%%%%%%%%%
% eTEMST_STEM=11, eTEMST_ISTEM=12, eTEMST_CBED=21, eTEMST_CBEI=22, eTEMST_ED=31, eTEMST_HRTEM=32, eTEMST_PED=41, eTEMST_HCTEM=42, eTEMST_EWFS=51, eTEMST_EWRS=52, 
% eTEMST_EELS=61, eTEMST_EFTEM=62, eTEMST_ProbeFS=71, eTEMST_ProbeRS=72, eTEMST_PPFS=81, eTEMST_PPRS=82,eTEMST_TFFS=91, eTEMST_TFRS=92
input_multislice.simulation_type = 32;

%%%%%%%%%%%%%% Electron-Specimen interaction model %%%%%%%%%%%%%%%%%
input_multislice.interaction_model = 1;              % eESIM_Multislice = 1, eESIM_Phase_Object = 2, eESIM_Weak_Phase_Object = 3
input_multislice.potential_type = 6;                 % ePT_Doyle_0_4 = 1, ePT_Peng_0_4 = 2, ePT_Peng_0_12 = 3, ePT_Kirkland_0_12 = 4, ePT_Weickenmeier_0_12 = 5, ePT_Lobato_0_12 = 6

%%%%%%%%%%%%%%%%%%%%%%% Potential slicing %%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.potential_slicing = 1;              % ePS_Planes = 1, ePS_dz_Proj = 2, ePS_dz_Sub = 3, ePS_Auto = 4

%%%%%%%%%%%%%%% Electron-Phonon interaction model %%%%%%%%%%%%%%%%%%
input_multislice.pn_model = 3;                       % ePM_Still_Atom = 1, ePM_Absorptive = 2, ePM_Frozen_Phonon = 3
input_multislice.pn_coh_contrib = 0;
input_multislice.pn_single_conf = 0;                 % 1: true, 0:false (extract single configuration)
input_multislice.pn_nconf = 10;                      % true: specific phonon configuration, false: number of frozen phonon configurations
input_multislice.pn_dim = 110;                       % phonon dimensions (xyz)
input_multislice.pn_seed = 300183;                   % Random seed(frozen phonon)

%%%%%%%%%%%%%%%%%%%%%%% Specimen information %%%%%%%%%%%%%%%%%%%%%%%
input_multislice.spec_atoms = [6, 5, 5, 1, 0.05, 1 ,0];
input_multislice.spec_lx= 10;
input_multislice.spec_ly=10;
input_multislice.spec_lz=10;
input_multislice.spec_dz=2;

fprintf('helloround%f\n', i)
csvmos2 = readtable(sprintf('/home/helen/Documents/HelenExperimentalCode/Defectivelatticegeneration/defectiveMoS2lattice_%.f.csv', i));
%csvmos2 = readtable('/home/helen/Documents/HelenExperimentalCode/Defectivelatticegeneration/defectiveMoS2lattice_1.csv');
mos2= table2array(csvmos2(:,:));
%the python output file has row numbers and column titles. The following
%lines get rid of this so that it is an appropriate MULTEM form
mos2(:,1) = [];
mos2(1,:) = [];

%box_size gets the max value so that lx and ly are automatically set to be
%appropriate for the input sample size- so the sample is in the middle of
%the vieiwng figure

box_size = max(mos2(:,2:3));
input_multislice.spec_atoms= mos2;
input_multislice.spec_lx=box_size(1);
input_multislice.spec_ly=box_size(2);
input_multislice.spec_lz=10;
input_multislice.spec_dz=2;

%%%%%%%%%%%%%%%%%%%%%% Specimen thickness %%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.thick_type = 1;                     % eTT_Whole_Spec = 1, eTT_Through_Thick = 2, eTT_Through_Slices = 3
%input_multislice.thick = c:c:1000;                   % Array of thickes (�)

%%%%%%%%%%%%%%%%%%%%%% x-y sampling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.nx = 1024;
input_multislice.ny = 1024;
input_multislice.bwl = 0;                            % Band-width limit, 1: true, 0:false

%%%%%%%%%%%%%%%%%%%% Microscope parameters %%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.E_0 = 300;                          % Acceleration Voltage (keV)
input_multislice.theta = 0.0;                        % Till ilumination (�)
input_multislice.phi = 0.0;                          % Till ilumination (�)

%%%%%%%%%%%%%%%%%%%%%% Illumination model %%%%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.illumination_model = 2;             % 1: coherente mode, 2: Partial coherente mode, 3: transmission cross coefficient, 4: Numerical integration
input_multislice.temporal_spatial_incoh = 1;         % 1: Temporal and Spatial, 2: Temporal, 3: Spatial

%%%%%%%%%%%%%%%%%%%%%%%% condenser lens %%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%% source spread function %%%%%%%%%%%%
ssf_sigma = il_mrad_2_sigma(input_multislice.E_0, 0.02);  % mrad to standard deviation
input_multislice.obj_lens_ssf_sigma = ssf_sigma;          % standard deviation: For parallel ilumination(�^-1); otherwise (�)
input_multislice.obj_lens_ssf_npoints = 4;                % # of integration points. It will be only used if illumination_model=4

%%%%%%%%%%%%%%%%%%%%%%%% Objective lens %%%%%%%%%%%%%%%%%%%%%%%%
input_multislice.obj_lens_m = 0;                   % Vortex momentum
input_multislice.obj_lens_c_10 = 20;               % Defocus (�)
input_multislice.obj_lens_c_30 = 0.04;             % Third order spherical aberration (mm)
input_multislice.obj_lens_c_50 = 0.00;             % Fifth order spherical aberration (mm)
input_multislice.obj_lens_c_12 = 0.0;              % Twofold astigmatism (�)
input_multislice.obj_lens_phi_12 = 0.0;            % Azimuthal angle of the twofold astigmatism (�)
input_multislice.obj_lens_c_23 = 0.0;              % Threefold astigmatism (�)
input_multislice.obj_lens_phi_23 = 0.0;            % Azimuthal angle of the threefold astigmatism (�)
input_multislice.obj_lens_inner_aper_ang = 0.0;    % Inner aperture (mrad) 
input_multislice.obj_lens_outer_aper_ang = 0.0;    % Outer aperture (mrad)

%%%%%%%%% defocus spread function %%%%%%%%%%%%
dsf_sigma = il_iehwgd_2_sigma(32); % from defocus spread to standard deviation
input_multislice.obj_lens_dsf_sigma = dsf_sigma;   % standard deviation (�)
input_multislice.obj_lens_dsf_npoints = 5;         % # of integration points. It will be only used if illumination_model=4

%%%%%%%%% zero defocus reference %%%%%%%%%%%%
input_multislice.obj_lens_zero_defocus_type = 1;   % eZDT_First = 1, eZDT_User_Define = 2
input_multislice.obj_lens_zero_defocus_plane = 0;

clear il_MULTEM;
tic;
output_multislice = il_MULTEM(system_conf, input_multislice); 
toc;

figure(1);
%for j=1:length(output_multislice.data)
j=length(output_multislice.data);
imagesc(output_multislice.data(j).m2psi_tot);
%image=output_multislice.data(j).m2psi_tot
title(strcat('Total intensity -  Thick = ', num2str(output_multislice.thick(j))));
axis image;
colormap gray;
saveas(figure(1), sprintf('/home/helen/Documents/HelenExperimentalCode/Defectivelatticegeneration/defectiveMoS2lattice_%.f.png',i))
%saveas(figure(1), '/home/helen/Documents/HelenExperimentalCode/Defectivelatticegeneration/defectiveMoS2lattice_1.png')
%imwrite(image ,sprintf('/home/helen/Documents/HelenExperimentalCode/Defectivelatticegeneration/defectiveMoS2lattice_0.png') )
%pause(0.25);
%end
count= (i)
end
defective_lastcount= output_multislice.data.m2psi_tot;