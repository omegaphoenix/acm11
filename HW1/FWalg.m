function distances = FWalg(adjacencyMat)
% FWALG Calculate shortest paths between all nodes using Floyd-Warshall
% shortest path algorithm

% Initialize D0
numNodes = length(adjacencyMat);
% Initialize everything to Infinity except the node to itself
distances = 1 ./ eye(numNodes) - 1;
% Add paths between adjacent nodes
distances = min(distances, adjacencyMat);

for k = 1:numNodes
  % Calculate distance of path passing through node k
  Dk = repmat(distances(:,k), 1, numNodes) ...
     + repmat(distances(k,:), numNodes, 1);
  % Take minimum distance using nodes 1...k-1 and k
  distances = min(distances, Dk);
end

end
