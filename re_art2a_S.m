%0.8, 0.7, 0.7
%[PIDCell, PIDCount, OutWM] = cluster_art2a(bipolar, 2, 250, 0.05, 0.8);
function [MCell, MCount, OWM, FraC] = re_art2a_S(Matrix, polarity, learning, Vigilance, InWM)

% Matrix = Matrix * 1000;
% Matrix(Matrix > 1) = log10(Matrix(Matrix > 1));
% polarity == 0 (neg) | == 1 (pos) | == 2 (both)
% Matrix(:,289) = 0;
tic;
if polarity == 0
    Matrix = Matrix(:,1:250); Base = sqrt(sum(Matrix .^2,2));
    Matrix = Matrix ./ repmat(Base,1,size(Matrix,2));
elseif polarity == 1
    Matrix = Matrix(:,251:500); Base = sqrt(sum(Matrix .^2,2));
    Matrix = Matrix ./ repmat(Base,1,size(Matrix,2));
else
    NegArea = Matrix(:,1:250); PosArea = Matrix(:,251:500);
    Base = sqrt(sum(NegArea .^2,2));
    NegArea = NegArea ./ repmat(Base,1,size(NegArea,2)) / sqrt(2);
    Base = sqrt(sum(PosArea .^2,2));
    PosArea = PosArea ./ repmat(Base,1,size(PosArea,2)) /sqrt(2);
    Matrix = [NegArea, PosArea];
end
clear NegArea PosArea Base;
toc;

NumSpectra = size(Matrix,1);
NumSeeds = 0;

if ~exist('InWM','var')
    Idx = randperm(NumSpectra);
    WM = Matrix(Idx(1),:);
elseif ~isempty(InWM)
    learning = 0;
    WM = InWM;
    NumSeeds = size(InWM,1);
end
WM = WM'; % use transpose of WM in main loop

Lambda = learning / (1 - learning);
Neuron = zeros(NumSpectra,1);
Proximity = zeros(NumSpectra,1);
FraC = zeros(20,1);
Iteration = 1;
StopCond = {'Iteration',20};

while 1
    fprintf('ART-2a Iteration %i\n',Iteration);
    LastNeuron = Neuron;
    
    tic;
    fprintf('step.1 calculation, depends on particle number\n');
    PermSpectra = randperm(NumSpectra);
    for i = 1:NumSpectra
        fprintf('Total = %i, Iteration = %i, i = %i.\n',NumSpectra,Iteration,i);
        j = PermSpectra(i);
        [Proximity(j), Neuron(j)] = max(Matrix(j,:) * WM);
        if Proximity(j) >= Vigilance
            if Neuron(j) > NumSeeds
                WM(:,Neuron(j)) = WM(:,Neuron(j)) + Matrix(j,:)' * Lambda;
                WM(:,Neuron(j)) = WM(:,Neuron(j)) / norm(WM(:,Neuron(j)));
            end
        else
            WM = [WM'; Matrix(j,:)]';
            Neuron(j) = size(WM,2);
        end
    end
    toc;
    
    tic;
    fprintf('step.2 remove empty neurons from WM\n');
    [C,~,IB] = intersect(Neuron,1:size(WM,2));
    WM = WM(:,IB);
    n = zeros(size(Neuron));
    for i = 1:length(C)
        n(Neuron == C(i)) = i;
    end
    Neuron = n;
    toc;
    
    FraC(Iteration) = sum(Neuron ~= LastNeuron) / NumSpectra; 
    
    % terminate loop
    Iteration = Iteration + 1;
    if Iteration > StopCond{2}
        warning('Reached MaxIteration');
        break;
    end
    
end

tic;
fprintf('step.3 fill into MCell\n');
% [SortNeuron, SortIdx] = sort(Neuron);
NumNeuron = size(WM,2);
MCell   = cell(NumNeuron,1);
MCount  = zeros(NumNeuron,1);
for i = 1:NumNeuron;
  Idx = find(Neuron == i); %Idx = find(SortNeuron == i);
  MCount(i) = length(Idx);
  MCell{i} = Idx; %MCell{i} = SortIdx(Idx);
end
toc;

tic;
fprintf('step.4 sort neurons based on population\n');
[~, SortIdx] = sort(MCount);
SortIdx = SortIdx(end:-1:1);
MCount = MCount(SortIdx);
MCell = MCell(SortIdx);
OWM = WM';
OWM = OWM(SortIdx,:);
toc;

return