for i= 0:9
    csvdata = readtable(sprintf('defectiveMoS2lattice_cellholes_%.f.csv', i), 'HeaderLines', 0);
    disp(sprintf('helloround%f',i))
    lattice= table2array(csvdata(:,:));
    figure1= scatter(10.24*lattice(1:end,2), 10*lattice(1:end,3));
    set(gca, 'YDir','reverse', 'color', 'None');
    colormap gray;
    axis image;
    xlim([0,1000]);
    ylim([0,1000]);
    saveas(figure1, sprintf('defectiveMoS2lattice_cellholes_%.f.png', i))
end
