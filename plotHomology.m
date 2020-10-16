function [f] = plotHomology(maxSizeA, maxSizeB)        

    Surf_locR = '/keilholz-lab/Jacob/TDABrains_00/data/null_WG33/S900.R.flat.32k_fs_LR.surf.gii'
    Surf_locL = '/keilholz-lab/Jacob/TDABrains_00/data/null_WG33/S900.L.flat.32k_fs_LR.surf.gii'

    %Surf_locR = ['/keilholz-lab/Jacob/TDABrains_00/data/null_WG33/S900.r.very_inflated_MSMAll.32k_fs_LR.surf.gii'];
    %Surf_locL = ['/keilholz-lab/Jacob/TDABrains_00/data/null_WG33/S900.L.very_inflated_MSMAll.32k_fs_LR.surf.gii'];

    GFL = gifti(Surf_locL);
    GFR = gifti(Surf_locR);
    
    gg = GFR.vertices*0;
    gg(:,1) = 500;
    GFR.vertices = GFR.vertices+gg;
    %vert = GFR.vertices;
    %X = x*cos(theta) - y*sin(theta);
    %Y = x*sin(theta) + y*cos(theta);
    
    mask_loc = ['/keilholz-lab/Jacob/TDABrains_00/data/null_WG33/Gordon333_FreesurferSubcortical.32k_fs_LR.dlabel.nii']
    Atlas = ft_read_cifti(mask_loc);
    
    VVL = Atlas.brainstructure == 1;
    VVR = Atlas.brainstructure == 2;

    VertexRegionsL = squeeze(Atlas.x333cort_subcort(VVL,:));
    VertexRegionsR = squeeze(Atlas.x333cort_subcort(VVR,:));
    
    % get centroids
    centroids = zeros(max([VertexRegionsL;VertexRegionsR]),2);

    h = unique(VertexRegionsL);
    h = h(~isnan(h));
    h = h(h>0);
    for i = transpose(h)
        voxels = Atlas.x333cort_subcort==i;
        Lv = voxels(VVL);    
        xs = double(GFL.vertices(Lv,1));
        ys = double(GFL.vertices(Lv,2)); 
        k = convhull(xs, ys);
        polyL = polyshape(xs(k),ys(k));
        [xi, yi] = centroid(polyL);
        j = knnsearch([xs,ys], [xi,yi]);
        centroids(i,:) = [xs(j), ys(j)];   
    end

    h = unique(VertexRegionsR);
    h = h(~isnan(h));    
    h = h(h>0);    
    for i = transpose(h)
        voxels = Atlas.x333cort_subcort==i;
        Rv = voxels(VVR);    
        xs = double(GFR.vertices(Rv,1));
        ys = double(GFR.vertices(Rv,2)); 
        k = convhull(xs, ys);
        polyR = polyshape(xs(k),ys(k));    
        [xi, yi] = centroid(polyR);
        j = knnsearch([xs,ys], [xi,yi]);
        centroids(i,:) = [xs(j), ys(j)];
    end

    missingVoxels = [133, 296, 299, 302, 304] + 1;
    centroids(missingVoxels,:) = [];
    
    f = figure('position',[10,10,1000,500]); clf;
    
    set(f,'color',[1,1,1]);
    set(f,'InvertHardcopy','off');
    f.PaperUnits = 'inches';
    f.PaperPosition = [0 0 16 9];   
    
    % Plot A) Homology More

    subplot(2,1,1);
    ax = gca
    axis tight off    
        
    bpL = patch('faces',GFL.faces,'vertices',GFL.vertices);        
    bpR = patch('faces',GFR.faces,'vertices',GFR.vertices);

    colorTransformFile = 'NodesMore.txt'
    colorTransform = importdata(colorTransformFile,',',0);        
    unqColors = length(unique(colorTransform));
    colorTransform(missingVoxels) = nan;
    
    vec = Atlas.x333cort_subcort(VVL,:);   
    bpL.FaceVertexCData = nan(length(vec),1);
    bpL.FaceVertexCData(vec>0) = colorTransform(vec(vec>0));
    vec = Atlas.x333cort_subcort(VVR,:);
    bpR.FaceVertexCData = nan(length(vec),1);
    bpR.FaceVertexCData(vec>0) = colorTransform(vec(vec>0));

    bpL.EdgeAlpha = 0.0;
    bpR.EdgeAlpha = 0.0;
    bpL.FaceColor = 'flat';
    bpR.FaceColor = 'flat';

    bpCMap = brewermap(unqColors+1, 'spectral');
    %bpCMap = parula(max(Atlas.x333cort_subcort)+1);
    colormap(gca,flipud(bpCMap));
    
    % do cycles
    
    cycleFile = 'EdgesMore.txt'
    cycles = importdata(cycleFile,',',0);
    
    szCyc = size(cycles)
    for i = 1:szCyc(1)
        points = [centroids(cycles(i,1),:);centroids(cycles(i,2),:)];            
        lineval = maxSizeB*cycles(i,3);
        lineA2 = line(points(:,1),points(:,2),'linewidth', lineval, 'linestyle','-','color',[.2,.6,.2,.7]);
    end
    colorbar
    
    cbar = colorbar;
    %cbar.Label.String = ['Clustering at threshold ', str(min(edges))]
    cbar.Label.String = 'Clustering at threshold'
    
    %legend(ax,[lineA2,lineB2],'Similar edge values','Contrasting edge values','Orientation','Horizontal','Location','southwest')
    
    ax.Title.String = {'Math Task, Diagram 1, More Correct Responses', ...
    'Edges show maximal H1 co-cycle || Nodes show clustering at max H1 threshold'}
        
    
    % Plot 2B) Simplex Brains

    subplot(2,1,2);
    ax = gca
    axis tight off    
        
    bpL = patch('faces',GFL.faces,'vertices',GFL.vertices);        
    bpR = patch('faces',GFR.faces,'vertices',GFR.vertices);

    colorTransformFile = 'NodesLess.txt'
    colorTransform = importdata(colorTransformFile,',',0);        
    unqColors = length(unique(colorTransform));
    colorTransform(missingVoxels) = nan;
    
    vec = Atlas.x333cort_subcort(VVL,:);   
    bpL.FaceVertexCData = nan(length(vec),1);
    bpL.FaceVertexCData(vec>0) = colorTransform(vec(vec>0));
    vec = Atlas.x333cort_subcort(VVR,:);
    bpR.FaceVertexCData = nan(length(vec),1);
    bpR.FaceVertexCData(vec>0) = colorTransform(vec(vec>0));

    bpL.EdgeAlpha = 0.0;
    bpR.EdgeAlpha = 0.0;
    bpL.FaceColor = 'flat';
    bpR.FaceColor = 'flat';

    bpCMap = brewermap(unqColors+1,'spectral');
    %bpCMap = parula(max(Atlas.x333cort_subcort)+1);
    colormap(gca,flipud(bpCMap));
    
    % do cycles
    
    cycleFile = 'EdgesLess.txt'
    cycles = importdata(cycleFile,',',0);
    
    szCyc = size(cycles)
    for i = 1:szCyc(1)
        points = [centroids(cycles(i,1),:);centroids(cycles(i,2),:)];            
        lineval = maxSizeB*cycles(i,3);
        lineA2 = line(points(:,1),points(:,2),'linewidth', lineval, 'linestyle','-','color',[.2,.6,.2,.7]);
    end
    colorbar
    
    cbar = colorbar;
    cbar.Label.String = 'Clustering at threshold'
    
    %legend(ax,[lineA2,lineB2],'Similar edge values','Contrasting edge values','Orientation','Horizontal','Location','southwest')
    
    ax.Title.String = {'Math Task, Diagram 1, Fewer Correct Responses', ...
    'Edges show maximal H1 co-cycle || Nodes show clustering at max H1 threshold'}
                   
    saveas(f, 'Figure5_Homology_Math.png')