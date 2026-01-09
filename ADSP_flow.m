%% MATLAB code to generate a simple flowchart

close all; clear; clc;

%% Define flowchart steps as STRING ARRAY (not cell)
steps = string([ ...
    "Start"
    "Prepare Audio Sources"
    "Develop Room Simulation and Mixing Function"
    "Generate 100 Mixed Sounds"
    "Convert Audio to Frequency Domain (STFT)"
    "Organize STFT Data into Datasets"
    "Implement Custom Data Generator"
    "Define Deep Learning Model Architecture"
    "Compile Deep Learning Model"
    "Train Deep Learning Model"
    "Evaluate Model and Reconstruct Audio"
    "Report Aggregate Metrics"
    "Provide Qualitative Audio Examples"
    "Final Task Summary"
    "End"
]);

%% Create directed graph
G = digraph();

%% Add nodes (NOW VALID)
G = addnode(G, steps);

%% Add edges
for i = 1:numel(steps)-1
    G = addedge(G, steps(i), steps(i+1));
end

%% Plot flowchart
figure('Name','Deep Learning Task Flowchart','NumberTitle','off');
h = plot(G, ...
    'Layout','layered', ...
    'Direction','down', ...
    'NodeFontSize',9, ...
    'MarkerSize',6);

%% Highlight Start and End
highlight(h,"Start",'NodeColor',[0 0.6 0],'MarkerSize',8,'LineWidth',1.5);
highlight(h,"End",'NodeColor',[0.7 0 0],'MarkerSize',8,'LineWidth',1.5);

title('Deep Learning Task Flowchart');
axis off;
