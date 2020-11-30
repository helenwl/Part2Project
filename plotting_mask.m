for i= 0:9
    csvdata = readtable(sprintf('defectiveMoS2lattice_cellholes_%.f.csv', i), 'HeaderLines', 0);
    disp(sprintf('helloround%f',i))
    lattice= table2array(csvdata(:,:));
    figure1= scatter(lattice(1:end,2), lattice(1:end,3));
    set(gca, 'YDir','reverse', 'xtick', [], 'ytick', [], 'color', 'None');
    colormap gray;
    axis image;
    xlim([0,100]);
    ylim([0,100]);
    saveas(figure1, sprintf('defectiveMoS2lattice_cellholes_%.f.png', i))
end
