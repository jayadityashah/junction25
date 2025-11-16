// Global state
let analysisData = null;
let currentRisk = null;
let network = null;

// Initialize the app
async function init() {
    await loadAnalysisData();
    showLandingPage();
    setupBackButton();
    setupTabs();
    setupChat();
}

// Load analysis JSON data (requirements-based)
async function loadAnalysisData() {
    try {
        const response = await fetch('requirements_analysis.json');
        if (!response.ok) {
            throw new Error('Requirements analysis file not found. Please run the pipeline first.');
        }
        analysisData = await response.json();
    } catch (error) {
        console.error('Error loading analysis data:', error);
        document.body.innerHTML = `
            <div style="padding: 2rem; text-align: center;">
                <h2>⚠️ Requirements Analysis Not Found</h2>
                <p>Please run the requirement extraction pipeline first:</p>
                <pre style="background: #f3f4f6; padding: 1rem; border-radius: 8px; margin: 1rem auto; max-width: 600px; text-align: left;">
# Step 1: Extract requirements
python risk_requirement_pipeline/requirement_extraction_pipeline.py

# Step 2: Find relationships
python risk_requirement_pipeline/requirement_relationship_pipeline.py

# Step 3: Export for frontend
python risk_requirement_pipeline/export_for_frontend.py
                </pre>
            </div>
        `;
        analysisData = { risk_categories: [] };
    }
}

// Show landing page with risk categories
function showLandingPage() {
    document.getElementById('landing-page').style.display = 'block';
    document.getElementById('graph-page').style.display = 'none';
    
    const container = document.getElementById('risk-categories');
    container.innerHTML = '';
    
    if (!analysisData.risk_categories || analysisData.risk_categories.length === 0) {
        container.innerHTML = '<p class="loading">No risk categories found</p>';
        return;
    }
    
    // Sort risk categories by total relationships (overlaps + contradictions) descending
    const sortedCategories = analysisData.risk_categories.sort((a, b) => {
        const aRelationships = (a.overlaps ? a.overlaps.length : 0) + (a.contradictions ? a.contradictions.length : 0);
        const bRelationships = (b.overlaps ? b.overlaps.length : 0) + (b.contradictions ? b.contradictions.length : 0);
        return bRelationships - aRelationships; // Descending order
    });

    sortedCategories.forEach((risk, index) => {
        const overlapCount = risk.overlaps ? risk.overlaps.length : 0;
        const contradictionCount = risk.contradictions ? risk.contradictions.length : 0;
        const requirementCount = risk.total_requirements || 0;

        // Calculate unique document count from overlaps and contradictions
        const allDocs = new Set();
        [...(risk.overlaps || []), ...(risk.contradictions || [])].forEach(item => {
            item.documents.forEach(doc => allDocs.add(doc.filename));
        });
        const docCount = allDocs.size;

        const card = document.createElement('div');
        card.className = 'risk-card';
        card.style.background = getGradientColor(index);
        card.onclick = () => showGraphView(risk, index);

        card.innerHTML = `
            <h2>${risk.risk_name}</h2>
            <p class="description">${risk.description}</p>
            <div class="metrics">
                <div class="metric">
                    <span class="metric-value">${requirementCount}</span>
                    <span class="metric-label">Requirements</span>
                </div>
                <div class="metric">
                    <span class="metric-value">${docCount}</span>
                    <span class="metric-label">Documents</span>
                </div>
                <div class="metric">
                    <span class="metric-value">${overlapCount}</span>
                    <span class="metric-label">Overlaps</span>
                </div>
                <div class="metric">
                    <span class="metric-value">${contradictionCount}</span>
                    <span class="metric-label">Contradictions</span>
                </div>
            </div>
        `;

        container.appendChild(card);
    });
}

// Cache for document metadata
const docMetadataCache = {};

// Get document display info from API
async function getDocInfo(filename) {
    // Check cache first
    if (docMetadataCache[filename]) {
        return docMetadataCache[filename];
    }
    
    try {
        const response = await fetch(`/api/document/metadata/${encodeURIComponent(filename)}`);
        if (response.ok) {
            const metadata = await response.json();
            docMetadataCache[filename] = metadata;
            return metadata;
        }
    } catch (error) {
        console.error('Error fetching metadata:', error);
    }
    
    // Fallback
    return {
        filename: filename,
        display_name: filename.replace('.di.json', ''),
        short_name: filename.replace('.di.json', '').substring(0, 20),
        pdf_path: null
    };
}

// Show graph view for a specific risk
async function showGraphView(risk, riskIndex) {
    currentRisk = risk;
    
    document.getElementById('landing-page').style.display = 'none';
    document.getElementById('graph-page').style.display = 'block';
    
    document.getElementById('graph-title').textContent = risk.risk_name;
    document.getElementById('graph-subtitle').textContent = risk.description;
    
    // Build and render network graph
    await buildNetworkGraph(risk);
}

// Find disconnected components in the graph using DFS
function findDisconnectedComponents(nodes, edges) {
    const allNodes = nodes.get();
    const allEdges = edges.get();
    
    // Build adjacency list
    const adjacency = {};
    allNodes.forEach(node => {
        adjacency[node.id] = [];
    });
    
    allEdges.forEach(edge => {
        adjacency[edge.from].push(edge.to);
        adjacency[edge.to].push(edge.from);
    });
    
    // Find components using DFS
    const visited = new Set();
    const components = [];
    
    function dfs(nodeId, component) {
        if (visited.has(nodeId)) return;
        visited.add(nodeId);
        component.push(nodeId);
        
        adjacency[nodeId].forEach(neighborId => {
            dfs(neighborId, component);
        });
    }
    
    allNodes.forEach(node => {
        if (!visited.has(node.id)) {
            const component = [];
            dfs(node.id, component);
            components.push(component);
        }
    });
    
    return components;
}

// Position disconnected components in separate regions to prevent overlap
function positionDisconnectedComponents(nodes, edges, components) {
    // Calculate grid layout based on number of components
    const numComponents = components.length;
    const cols = Math.ceil(Math.sqrt(numComponents));
    const rows = Math.ceil(numComponents / cols);
    
    // Spacing between component centers - increased dramatically
    const spacingX = 1500;
    const spacingY = 1200;
    
    // Get all edges
    const allEdges = edges.get();
    
    components.forEach((component, idx) => {
        const row = Math.floor(idx / cols);
        const col = idx % cols;
        
        // Calculate center position for this component
        const centerX = (col - (cols - 1) / 2) * spacingX;
        const centerY = (row - (rows - 1) / 2) * spacingY;
        
        // Separate hubs from document nodes
        const hubs = [];
        const docs = [];
        
        component.forEach(nodeId => {
            const node = nodes.get(nodeId);
            if (node && node.isHub) {
                hubs.push(nodeId);
            } else {
                docs.push(nodeId);
            }
        });
        
        if (hubs.length > 0) {
            // Component has hub(s) - position each hub individually in a grid
            hubs.forEach((hubId, hubIdx) => {
                // Calculate position for each hub in a sub-grid
                const hubCols = Math.ceil(Math.sqrt(hubs.length));
                const hubRows = Math.ceil(hubs.length / hubCols);
                const hubRow = Math.floor(hubIdx / hubCols);
                const hubCol = hubIdx % hubCols;

                // Position hubs in a tighter grid within the component area
                const hubSpacingX = spacingX / Math.max(hubCols, 2);
                const hubSpacingY = spacingY / Math.max(hubRows, 2);
                const hubX = centerX + (hubCol - (hubCols - 1) / 2) * hubSpacingX;
                const hubY = centerY + (hubRow - (hubRows - 1) / 2) * hubSpacingY;

                nodes.update({
                    id: hubId,
                    x: hubX,
                    y: hubY,
                    physics: {
                        enabled: false  // Disable physics for hubs to keep them in position
                    }
                });
            });

            // Position document nodes in a circle around the component center
            // (simplified approach - all docs in this component go around the center)
            docs.forEach((nodeId, nodeIdx) => {
                const angle = (2 * Math.PI * nodeIdx) / docs.length;
                const radius = Math.max(200, docs.length * 30);

                nodes.update({
                    id: nodeId,
                    x: centerX + radius * Math.cos(angle),
                    y: centerY + radius * Math.sin(angle),
                    physics: {
                        enabled: true
                    }
                });
            });
        } else {
            // No hub - simple circular layout
            component.forEach((nodeId, nodeIdx) => {
                const angle = (2 * Math.PI * nodeIdx) / component.length;
                const radius = Math.max(150, component.length * 25);
                
                nodes.update({
                    id: nodeId,
                    x: centerX + radius * Math.cos(angle),
                    y: centerY + radius * Math.sin(angle),
                    physics: {
                        enabled: true
                    }
                });
            });
        }
    });
}

// Build network graph
async function buildNetworkGraph(risk) {
    const container = document.getElementById('network');
    
    const nodes = new vis.DataSet();
    const edges = new vis.DataSet();
    
    // Get list of all unique document filenames
    const allFilenames = new Set();
    
    [...(risk.overlaps || []), ...(risk.contradictions || [])].forEach(item => {
        item.documents.forEach(doc => {
            allFilenames.add(doc.filename);
        });
    });
    
    // Add document nodes (fetch metadata for each)
    for (const filename of allFilenames) {
        const docInfo = await getDocInfo(filename);
        nodes.add({
            id: filename,
            label: docInfo.short_name,
            title: docInfo.display_name,
            shape: 'box',
            margin: 12,
            font: { size: 14, face: 'arial', bold: true },
            color: {
                background: '#ffffff',
                border: '#2563eb',
                highlight: {
                    background: '#dbeafe',
                    border: '#1d4ed8'
                }
            },
            borderWidth: 2,
            borderWidthSelected: 3
        });
    }
    
    // Track edges between node pairs for curve separation
    const edgesBetweenPairs = new Map();
    
    // First pass: collect all edges between pairs
    const allEdgesToAdd = [];
    
    // Collect overlap edges
    if (risk.overlaps) {
        risk.overlaps.forEach((overlap, overlapIndex) => {
            const docFilenames = overlap.documents.map(d => d.filename);
            
            if (docFilenames.length === 2) {
                const pairKey = [docFilenames[0], docFilenames[1]].sort().join('|');
                if (!edgesBetweenPairs.has(pairKey)) {
                    edgesBetweenPairs.set(pairKey, []);
                }
                allEdgesToAdd.push({
                    id: overlap.id || `overlap-${overlapIndex}`,
                    from: docFilenames[0],
                    to: docFilenames[1],
                    color: { color: '#3b82f6', highlight: '#1d4ed8' },
                    width: 3,
                    type: 'overlap',
                    pairKey: pairKey,
                    data: {
                        type: 'overlap',
                        overlap: overlap,
                        riskName: risk.risk_name
                    }
                });
                edgesBetweenPairs.get(pairKey).push(allEdgesToAdd[allEdgesToAdd.length - 1]);
            }
        });
    }
    
    // Collect contradiction edges
    if (risk.contradictions) {
        risk.contradictions.forEach((contradiction, contradictionIndex) => {
            const docFilenames = contradiction.documents.map(d => d.filename);
            
            if (docFilenames.length === 2) {
                const pairKey = [docFilenames[0], docFilenames[1]].sort().join('|');
                if (!edgesBetweenPairs.has(pairKey)) {
                    edgesBetweenPairs.set(pairKey, []);
                }
                allEdgesToAdd.push({
                    id: contradiction.id || `contradiction-${contradictionIndex}`,
                    from: docFilenames[0],
                    to: docFilenames[1],
                    color: { color: '#ef4444', highlight: '#dc2626' },
                    width: 4,
                    dashes: [10, 10],
                    type: 'contradiction',
                    pairKey: pairKey,
                    data: {
                        type: 'contradiction',
                        contradiction: contradiction,
                        riskName: risk.risk_name
                    }
                });
                edgesBetweenPairs.get(pairKey).push(allEdgesToAdd[allEdgesToAdd.length - 1]);
            }
        });
    }
    
    // Second pass: add edges with proper curve separation
    allEdgesToAdd.forEach(edge => {
        const edgesInPair = edgesBetweenPairs.get(edge.pairKey);
        const edgeIndex = edgesInPair.indexOf(edge);
        const totalEdges = edgesInPair.length;
        
        let smoothConfig;
        if (totalEdges === 1) {
            // Single edge - straight or gentle curve
            smoothConfig = { type: 'continuous', roundness: 0.2 };
        } else {
            // Multiple edges - spread them out with different roundness
            // Calculate roundness to spread edges evenly
            const baseRoundness = 0.3;
            const step = 0.25;
            const offset = (edgeIndex - (totalEdges - 1) / 2) * step;
            smoothConfig = {
                type: 'curvedCW',
                roundness: baseRoundness + offset
            };
        }
        
        edges.add({
            id: edge.id,
            from: edge.from,
            to: edge.to,
            color: edge.color,
            width: edge.width,
            dashes: edge.dashes,
            smooth: smoothConfig,
            data: edge.data
        });
    });
    
    // Add overlap edges with hubs (multi-document)
    if (risk.overlaps) {
        risk.overlaps.forEach((overlap, overlapIndex) => {
            const docFilenames = overlap.documents.map(d => d.filename);
            
            if (docFilenames.length > 2) {
                // Multi-document: create invisible hub and curved edges converging to it
                const hubId = `hub-overlap-${overlapIndex}`;
                
                // Create invisible hub node at the center
                nodes.add({
                    id: hubId,
                    label: '',
                    shape: 'dot',
                    size: 25,  // Invisible hit area for clicking (30% smaller)
                    color: {
                        background: 'transparent',  // Completely transparent
                        border: 'transparent',
                        hover: {
                            background: 'transparent',  // Stay transparent on hover
                            border: 'transparent'
                        },
                        highlight: {
                            background: 'transparent',  // Stay transparent on highlight
                            border: 'transparent'
                        }
                    },
                    physics: {
                        enabled: true
                    },
                    mass: 10,  // Very heavy to act as anchor point for connected nodes
                    fixed: false,
                    title: `${docFilenames.length}-way overlap - click to view`,
                    isHub: true,
                    hubData: {
                        type: 'overlap',
                        overlap: overlap
                    }
                });
                
                // Connect each document to the invisible hub with smooth curves
                docFilenames.forEach((filename, idx) => {
                    const edgeId = `${overlap.id || `overlap-${overlapIndex}`}-${idx}`;
                    edges.add({
                        id: edgeId,
                        from: filename,
                        to: hubId,
                        color: { color: '#3b82f6', highlight: '#1d4ed8' },
                        width: 3,
                        smooth: {
                            enabled: true,
                            type: 'curvedCW',
                            roundness: 0.2
                        },
                        length: 250,  // Shorter edges to hubs
                        groupId: overlap.id || `overlap-${overlapIndex}`,
                        data: {
                            type: 'overlap',
                            overlap: overlap,
                            riskName: risk.risk_name
                        }
                    });
                });
            }
        });
    }
    
    // Add contradiction edges with hubs (multi-document)
    if (risk.contradictions) {
        risk.contradictions.forEach((contradiction, contradictionIndex) => {
            const docFilenames = contradiction.documents.map(d => d.filename);
            
            if (docFilenames.length > 2) {
                // Multi-document: create invisible hub and curved edges converging to it
                const hubId = `hub-contradiction-${contradictionIndex}`;
                
                // Create invisible hub node at the center
                nodes.add({
                    id: hubId,
                    label: '',
                    shape: 'dot',
                    size: 25,  // Invisible hit area for clicking (30% smaller)
                    color: {
                        background: 'transparent',  // Completely transparent
                        border: 'transparent',
                        hover: {
                            background: 'transparent',  // Stay transparent on hover
                            border: 'transparent'
                        },
                        highlight: {
                            background: 'transparent',  // Stay transparent on highlight
                            border: 'transparent'
                        }
                    },
                    physics: {
                        enabled: true
                    },
                    mass: 10,  // Very heavy to act as anchor point for connected nodes
                    fixed: false,
                    title: `${docFilenames.length}-way contradiction - click to view`,
                    isHub: true,
                    hubData: {
                        type: 'contradiction',
                        contradiction: contradiction
                    }
                });
                
                // Connect each document to the invisible hub with smooth curves
                docFilenames.forEach((filename, idx) => {
                    const edgeId = `${contradiction.id || `contradiction-${contradictionIndex}`}-${idx}`;
                    edges.add({
                        id: edgeId,
                        from: filename,
                        to: hubId,
                        color: { color: '#ef4444', highlight: '#dc2626' },
                        width: 4,
                        dashes: [10, 10],
                        smooth: {
                            enabled: true,
                            type: 'curvedCW',
                            roundness: 0.2
                        },
                        length: 250,  // Shorter edges to hubs
                        groupId: contradiction.id || `contradiction-${contradictionIndex}`,
                        data: {
                            type: 'contradiction',
                            contradiction: contradiction,
                            riskName: risk.risk_name
                        }
                    });
                });
            }
        });
    }
    
    // Detect disconnected components and position them with good initial layout
    const components = findDisconnectedComponents(nodes, edges);
    if (components.length >= 1) {
        positionDisconnectedComponents(nodes, edges, components);
    }
    
    const data = { nodes, edges };
    
    const options = {
        nodes: {
            shape: 'box',
            margin: 20,
            widthConstraint: { maximum: 200 },
            heightConstraint: { minimum: 40 },
            mass: 3  // Heavier to push edges away
        },
        edges: {
            smooth: {
                enabled: true,
                type: 'continuous',
                roundness: 0.2
            },
            arrows: { to: false },
            hoverWidth: 0,
            length: 300  // Longer edges to reduce crossing area
        },
        physics: {
            enabled: true,
            barnesHut: {
                gravitationalConstant: -80000,  // Even stronger repulsion
                centralGravity: 0.0,  // No central gravity
                springLength: 300,  // Longer spring length
                springConstant: 0.01,  // Weaker springs to maintain separation
                damping: 0.5,  // Higher damping for stability
                avoidOverlap: 1  // Maximum overlap avoidance
            },
            stabilization: {
                enabled: true,
                iterations: 1500,  // More iterations for better stabilization
                updateInterval: 25,
                fit: true
            },
            solver: 'barnesHut',
            minVelocity: 0.1,  // Lower threshold for finer stabilization
            adaptiveTimestep: true  // Better responsiveness during interaction
        },
        interaction: {
            hover: true,
            tooltipDelay: 100,
            dragNodes: true,
            dragView: true,
            zoomView: true,
            hideEdgesOnDrag: false,
            hideNodesOnDrag: false
        },
        layout: {
            improvedLayout: true,
            hierarchical: false
        }
    };
    
    network = new vis.Network(container, data, options);
    
    // Fit the view immediately with padding
    network.fit({
        animation: false,
        padding: 100  // Add 100px padding around the graph
    });
    
    // Ensure the view fits all nodes after stabilization with nice animation
    network.once('stabilizationIterationsDone', function() {
        network.fit({
            animation: {
                duration: 600,
                easingFunction: 'easeInOutQuad'
            },
            padding: 100  // Maintain 100px padding
        });
    });
    
    // Event handlers
    network.on('click', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const node = nodes.get(nodeId);
            
            // Check if it's an invisible hub
            if (node.isHub) {
                const hubData = node.hubData;
                const item = hubData.type === 'overlap' ? hubData.overlap : hubData.contradiction;
                showEdgeDetailsFromData(item, hubData.type);
            } else {
                showNodeDetails(nodeId, risk);
            }
        } else if (params.edges.length > 0) {
            const edgeId = params.edges[0];
            const edge = edges.get(edgeId);
            
            if (edge.data) {
                showEdgeDetails(edge);
            }
        }
    });
    
    // Highlight all edges in the same group on hover (edge or hub node)
    network.on('hoverEdge', function(params) {
        const edgeId = params.edge;
        const edge = edges.get(edgeId);
        
        if (edge.groupId) {
            highlightEdgeGroup(edge.groupId, edge.data.type, edges);
        }
    });
    
    network.on('blurEdge', function(params) {
        const edgeId = params.edge;
        const edge = edges.get(edgeId);
        
        if (edge.groupId) {
            resetEdgeGroup(edge.groupId, edge.data.type, edges);
        }
    });
    
    // Highlight edges when hovering over hub node
    network.on('hoverNode', function(params) {
        const nodeId = params.node;
        const node = nodes.get(nodeId);
        
        if (node.isHub) {
            // Find all edges connected to this hub
            const connectedEdges = network.getConnectedEdges(nodeId);
            if (connectedEdges.length > 0) {
                const firstEdge = edges.get(connectedEdges[0]);
                if (firstEdge.groupId) {
                    highlightEdgeGroup(firstEdge.groupId, firstEdge.data.type, edges);
                }
            }
        }
    });
    
    network.on('blurNode', function(params) {
        const nodeId = params.node;
        const node = nodes.get(nodeId);
        
        if (node.isHub) {
            // Find all edges connected to this hub
            const connectedEdges = network.getConnectedEdges(nodeId);
            if (connectedEdges.length > 0) {
                const firstEdge = edges.get(connectedEdges[0]);
                if (firstEdge.groupId) {
                    resetEdgeGroup(firstEdge.groupId, firstEdge.data.type, edges);
                }
            }
        }
    });
}

// Helper: Highlight all edges in a group uniformly
function highlightEdgeGroup(groupId, type, edges) {
    const groupEdges = edges.get({
        filter: function(item) {
            return item.groupId === groupId;
        }
    });
    
    const highlightColor = type === 'overlap' ? '#1d4ed8' : '#dc2626';
    const highlightWidth = type === 'overlap' ? 5 : 6;
    
    groupEdges.forEach(groupEdge => {
        edges.update({
            id: groupEdge.id,
            width: highlightWidth,
            color: { 
                color: highlightColor,
                highlight: highlightColor
            }
        });
    });
}

// Helper: Reset all edges in a group
function resetEdgeGroup(groupId, type, edges) {
    const groupEdges = edges.get({
        filter: function(item) {
            return item.groupId === groupId;
        }
    });
    
    const normalColor = type === 'overlap' ? '#3b82f6' : '#ef4444';
    const normalWidth = type === 'overlap' ? 3 : 4;
    
    groupEdges.forEach(groupEdge => {
        edges.update({
            id: groupEdge.id,
            width: normalWidth,
            color: { 
                color: normalColor,
                highlight: type === 'overlap' ? '#1d4ed8' : '#dc2626'
            }
        });
    });
}

// Show node (document) details
async function showNodeDetails(filename, risk) {
    const panel = document.getElementById('detail-panel');
    const content = document.getElementById('panel-content');
    
    const docInfo = await getDocInfo(filename);
    
    // Find all connections
    const connections = [];
    
    if (risk.overlaps) {
        for (const overlap of risk.overlaps) {
            const docs = overlap.documents.map(d => d.filename);
            if (docs.includes(filename)) {
                const otherDocs = overlap.documents.filter(d => d.filename !== filename);
                const otherNames = [];
                for (const doc of otherDocs) {
                    const info = await getDocInfo(doc.filename);
                    otherNames.push(info.display_name);
                }
                connections.push({
                    type: 'overlap',
                    others: otherNames,
                    data: overlap
                });
            }
        }
    }
    
    if (risk.contradictions) {
        for (const contradiction of risk.contradictions) {
            const docs = contradiction.documents.map(d => d.filename);
            if (docs.includes(filename)) {
                const otherDocs = contradiction.documents.filter(d => d.filename !== filename);
                const otherNames = [];
                for (const doc of otherDocs) {
                    const info = await getDocInfo(doc.filename);
                    otherNames.push(info.display_name);
                }
                connections.push({
                    type: 'contradiction',
                    others: otherNames,
                    data: contradiction
                });
            }
        }
    }
    
    content.innerHTML = `
        <h2>${docInfo.display_name}</h2>
        ${docInfo.pdf_path ? `<a href="/${docInfo.pdf_path}" target="_blank" class="pdf-button">📄 Open Full PDF</a>` : ''}
        
        <h3>Connections (${connections.length})</h3>
        <div class="connections-list">
            ${connections.length === 0 ? '<p class="loading">No connections for this document</p>' : ''}
            ${connections.map(conn => `
                <div class="connection-item ${conn.type}" onclick="showConnectionDetails(${JSON.stringify(conn.data).replace(/"/g, '&quot;')}, '${conn.type}')">
                    <div class="connection-header">
                        <span class="connection-type">${conn.type === 'overlap' ? '🔗 Overlap' : '⚠️ Contradiction'}</span>
                        <span>with ${conn.others.length} document${conn.others.length !== 1 ? 's' : ''}</span>
                    </div>
                    <p class="connection-others">${conn.others.join(', ')}</p>
                    <p class="connection-reason">${conn.data.reason.substring(0, 150)}...</p>
                    <button class="view-paragraphs-btn">View Requirements →</button>
                </div>
            `).join('')}
        </div>
    `;
    
    panel.classList.add('open');
}

// Make showConnectionDetails global
window.showConnectionDetails = function(connectionData, type) {
    showEdgeDetailsFromData(connectionData, type);
};

// Show edge details from connection data (requirements-based)
async function showEdgeDetailsFromData(item, type) {
    const panel = document.getElementById('detail-panel');
    const content = document.getElementById('panel-content');
    
    content.innerHTML = `
        <div class="edge-header">
            <span class="type-badge ${type}">
                ${type === 'overlap' ? '🔗 Overlap' : '⚠️ Contradiction'}
            </span>
            <h2>${item.documents.length} Documents ${type === 'overlap' ? 'Overlap' : 'Contradict'}</h2>
        </div>
        
        <div class="edge-reason">
            <h3>Analysis</h3>
            <p>${item.reason}</p>
        </div>
        
        <h3>Relevant Requirements</h3>
        <div id="content-loading" class="loading">Loading requirements...</div>
        <div id="content-container"></div>
    `;
    
    panel.classList.add('open');
    
    const contentContainer = document.getElementById('content-container');
    const loadingDiv = document.getElementById('content-loading');
    
    for (const docInfo of item.documents) {
        const metadata = await getDocInfo(docInfo.filename);
        
        const docSection = document.createElement('div');
        docSection.className = 'document-section';
        
        // Display requirements
        const requirements = docInfo.requirements || [];
        docSection.innerHTML = `
            <div class="doc-section-header">
                <h4>${metadata.display_name}</h4>
                ${metadata.pdf_path ? `<a href="/${metadata.pdf_path}" target="_blank" class="pdf-link-small">📄 PDF</a>` : ''}
            </div>
            ${requirements.length === 0 ? '<p class="loading">No requirements found</p>' : ''}
            ${requirements.map(req => `
                <div class="requirement-highlight">
                    <div class="requirement-meta">
                        <span class="page-tag">Pages ${req.start_page}-${req.end_page}</span>
                        ${req.risk_category ? `<span class="risk-tag">${req.risk_category.replace(/_/g, ' ')}</span>` : ''}
                    </div>
                    <div class="requirement-text">${req.text}</div>
                </div>
            `).join('')}
        `;
        
        contentContainer.appendChild(docSection);
    }
    
    loadingDiv.remove();
}

// Show edge (overlap/contradiction) details
async function showEdgeDetails(edge) {
    const edgeData = edge.data;
    const item = edgeData.type === 'overlap' ? edgeData.overlap : edgeData.contradiction;
    
    await showEdgeDetailsFromData(item, edgeData.type);
}

// Load requirements for a document (if needed for future features)
async function loadDocumentRequirements(filename) {
    try {
        const response = await fetch(`/api/requirements/${encodeURIComponent(filename)}`);
        
        if (!response.ok) {
            console.error('Error fetching requirements:', filename);
            return [];
        }
        
        return await response.json();
    } catch (error) {
        console.error('Error loading requirements:', error);
        return [];
    }
}

// Setup back button
function setupBackButton() {
    document.getElementById('back-button').onclick = () => {
        document.getElementById('detail-panel').classList.remove('open');
        showLandingPage();
    };
    
    document.getElementById('close-panel').onclick = () => {
        document.getElementById('detail-panel').classList.remove('open');
    };
}

// Utility: Get gradient color
function getGradientColor(index) {
    const gradients = [
        'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
        'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
        'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
    ];
    return gradients[index % gradients.length];
}

// Tab switching
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tab = button.dataset.tab;
            tabButtons.forEach(b => b.classList.remove('active'));
            button.classList.add('active');
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
                content.style.display = 'none';
            });
            const activeTab = document.getElementById(`${tab}-tab`);
            activeTab.classList.add('active');
            activeTab.style.display = 'block';
        });
    });
}

// Chat functionality
function setupChat() {
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('chat-send');
    const chatMessages = document.getElementById('chat-messages');
    let isProcessing = false;

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message || isProcessing) return;

        // Set processing state
        isProcessing = true;
        chatInput.disabled = true;
        chatSend.disabled = true;
        chatSend.classList.add('loading');
        chatSend.textContent = '';

        addUserMessage(message);
        chatInput.value = '';

        const loadingMsg = addLoadingMessage();

        try {
            const response = await fetch('/api/graphrag/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, method: 'local' })
            });

            const data = await response.json();
            removeLoadingMessage(loadingMsg);

            if (data.success) {
                addBotMessage(data.response);
            } else {
                addBotMessage('Error: ' + (data.error || 'Query failed'));
            }
        } catch (error) {
            removeLoadingMessage(loadingMsg);
            addBotMessage('Error: ' + error.message);
        } finally {
            // Reset processing state
            isProcessing = false;
            chatInput.disabled = false;
            chatSend.disabled = false;
            chatSend.classList.remove('loading');
            chatSend.textContent = 'Send';
            chatInput.focus();
        }
    }

    chatSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !isProcessing) {
            e.preventDefault();
            sendMessage();
        }
    });

    function addUserMessage(text) {
        const msg = document.createElement('div');
        msg.className = 'chat-message user';
        msg.innerHTML = `<div class="message-content"><p>${escapeHtml(text)}</p></div>`;
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addBotMessage(text) {
        const msg = document.createElement('div');
        msg.className = 'chat-message bot';
        msg.innerHTML = `<div class="message-content">${formatBotMessage(text)}</div>`;
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addLoadingMessage() {
        const msg = document.createElement('div');
        msg.className = 'chat-message bot loading-message';
        msg.innerHTML = '<div class="message-content"><div class="loading-dots"><span></span><span></span><span></span></div></div>';
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msg;
    }

    function removeLoadingMessage(msg) {
        if (msg) msg.remove();
    }

    function formatBotMessage(text) {
        return escapeHtml(text).replace(/\n/g, '<br>');
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
