
clear all;
close all; 

nDim = 1;
nSet = 3;
nProp = 3;
nPlotW = 2;
axisLabel = ["$x$ [m]"];
dataLabel = ["$P$ [Pa]","$u$ [m/s]","$T$ [K]"];
legLabel = ["Matlab","CHAMPS","Exact"];
file  = ["/home/tyler/Research/CHAMPS/GFM/debug/flowReimannExact_00000800.d" ...
         ,"/home/tyler/Research/CHAMPS/GFM/debug/flowReimannComputed_00000800.d" ...
         ,"/home/tyler/Research/CHAMPS/GFM/debug/flowReimannMatlab_00000800.d"
        ];
nHdrLine = [0,0,1];
nCol = [4,4,4];
dataCol = [1,2,3,4 ...
          ; 1,2,3,4 ...
          ; 1,2,3,4];

fprintf('[I] Plotting properties for %i data sets \n',nSet);

nPlotH = floor((nSet+2)/nPlotW);
figure
fig = tiledlayout(nPlotH,nPlotW);
hold on

for set = 1:nSet
    fprintf('[I] Reading file #%i = %s \n',set,file(set));
    fileID(set) = fopen(file(set),'r');
    for ln = 1:nHdrLine(set)
        dum = fgets(fileID(set));
    end    
    dataLoc = fscanf(fileID(set),'%f',[nCol(set),Inf]);
    dataIn(set,1:size(dataLoc,1),1:size(dataLoc,2)) = sort(dataLoc,dataCol(set,1));
    fprintf('[I] Data array size = %i x %i \n',size(dataLoc));
    for prop = 1:nProp+1
        data(set,prop,1:size(dataIn,3)) = dataIn(set,dataCol(set,prop),1:size(dataIn,3));
    end
end

txt_x = xlabel(axisLabel(1)); 
txt_x.Interpreter = 'latex'; 
for prop = 1:nProp
    nexttile(prop);
    txt_y = ylabel(dataLabel(prop)); 
    txt_y.Interpreter = 'latex'; 
    cla;
end

for prop = 1:nProp
    nexttile(prop);
    for set = 1:nSet
       x = squeeze(data(set,1,:));
       f = squeeze(data(set,prop+1,:));
       hold on;
       plot(x,f);
   end
end

% for set = 1:nSet
%     dataLoc = squeeze(data(set,:,:));
%     for prop = 1:nProp
%        nexttile(prop);
%        x = squeeze(data(set,prop,:));
%        plot(x,dataLoc(1,:));
%     end
% end

% for prop = 1:1
%     nexttile(prop);
%     dataLoc = squeeze(data(:,prop,:))
%     for set = 1:nSet
%         x = squeeze(data(set,1,:));
%         plot(x,dataLoc(set,:));
%         hold on;
%     end
% end

% for prop = 1:nProp
%     % dataLoc = squeeze(data(:,prop,:));
% end

% figure;
% for set = 1:nSet
%     dataLoc = squeeze(data(set,:,:));
%     prop = 1;
%     x = squeeze(data(set,1,:));
%     plot(x,dataLoc(set,:));
% end

