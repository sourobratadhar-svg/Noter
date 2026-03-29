/**
 * Knowledge Graph Screen
 * =====================
 * Visualizes relationships between note chunks using a force-directed graph.
 * Nodes = chunks, Edges = cosine similarity above threshold.
 * Uses WebView with d3-force for interactive zoom/pan/highlight.
 * Read-only view — no editing.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';

const API_URL = process.env.EXPO_PUBLIC_BACKEND_URL;

type GraphNode = {
  id: string;
  label: string;
  note_title: string;
  note_id: string;
  chunk_index: number;
  text_preview: string;
};

type GraphEdge = {
  source: string;
  target: string;
  weight: number;
};

type GraphData = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  threshold: number;
  cached: boolean;
};

/**
 * Generate the full HTML+JS for the d3-force graph rendered inside WebView.
 * Self-contained — includes d3 from CDN and all rendering logic.
 */
function buildGraphHtml(data: GraphData): string {
  const nodes = JSON.stringify(data.nodes.map(n => ({
    id: n.id,
    label: n.label,
    noteTitle: n.note_title,
    noteId: n.note_id,
    preview: n.text_preview,
  })));

  const edges = JSON.stringify(data.edges.map(e => ({
    source: e.source,
    target: e.target,
    weight: e.weight,
  })));

  return `<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { 
    background: #FFFFFF; overflow: hidden; 
    font-family: 'Courier New', monospace;
  }
  svg { width: 100vw; height: 100vh; display: block; }
  .tooltip {
    position: absolute; background: #0A0A0A; color: #FFF;
    padding: 10px 14px; font-size: 11px; font-family: 'Courier New', monospace;
    pointer-events: none; opacity: 0; transition: opacity 0.15s;
    max-width: 280px; line-height: 1.5; letter-spacing: 0.5px;
    border: 2px solid #002FA7;
  }
  .empty-msg {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    text-align: center; font-family: 'Courier New', monospace; color: #555;
  }
  .empty-msg h2 { font-size: 20px; color: #0A0A0A; letter-spacing: 2px; margin-bottom: 8px; }
  .empty-msg p { font-size: 12px; letter-spacing: 1px; }
</style>
</head>
<body>
<div class="tooltip" id="tooltip"></div>
<svg id="graph"></svg>
<script>
const nodes = ${nodes};
const links = ${edges};

if (nodes.length === 0) {
  document.body.innerHTML = '<div class="empty-msg"><h2>NO GRAPH DATA</h2><p>ADD MORE NOTES TO BUILD THE KNOWLEDGE GRAPH</p></div>';
} else {
  const svg = d3.select("#graph");
  const width = window.innerWidth;
  const height = window.innerHeight;
  const tooltip = d3.select("#tooltip");

  // Color scale: group nodes by noteId for consistent coloring
  const noteIds = [...new Set(nodes.map(n => n.noteId))];
  const colors = ['#002FA7','#FF2A00','#00D154','#FFDE00','#0A0A0A','#8B5CF6','#F97316','#06B6D4','#EC4899','#10B981'];
  const colorMap = {};
  noteIds.forEach((id, i) => { colorMap[id] = colors[i % colors.length]; });

  // Setup zoom
  const g = svg.append("g");
  const zoom = d3.zoom()
    .scaleExtent([0.2, 5])
    .on("zoom", (event) => g.attr("transform", event.transform));
  svg.call(zoom);

  // Force simulation
  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(d => 120 / (d.weight || 0.5)).strength(d => d.weight))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(30));

  // Draw edges
  const link = g.append("g").selectAll("line")
    .data(links).enter().append("line")
    .attr("stroke", "#CCCCCC")
    .attr("stroke-width", d => Math.max(1, d.weight * 4))
    .attr("stroke-opacity", 0.6);

  // Edge weight labels
  const linkLabel = g.append("g").selectAll("text")
    .data(links).enter().append("text")
    .text(d => d.weight.toFixed(2))
    .attr("font-size", "9px")
    .attr("font-family", "Courier New, monospace")
    .attr("fill", "#999")
    .attr("text-anchor", "middle");

  // Draw nodes
  const node = g.append("g").selectAll("g")
    .data(nodes).enter().append("g")
    .call(d3.drag()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      })
    );

  // Node circles
  node.append("circle")
    .attr("r", 14)
    .attr("fill", d => colorMap[d.noteId])
    .attr("stroke", "#0A0A0A")
    .attr("stroke-width", 2)
    .attr("cursor", "pointer");

  // Node labels (short)
  node.append("text")
    .text(d => {
      const parts = d.label.split('[');
      const name = parts[0];
      return name.length > 12 ? name.slice(0, 12) + '..' : name;
    })
    .attr("dx", 18).attr("dy", 4)
    .attr("font-size", "10px")
    .attr("font-family", "Courier New, monospace")
    .attr("fill", "#0A0A0A")
    .attr("font-weight", "bold");

  // Hover tooltip
  node.on("mouseenter", (event, d) => {
    tooltip.style("opacity", 1)
      .html("SRC // " + d.noteTitle + "<br><br>" + d.preview);
    // Highlight connected edges
    link.attr("stroke", l => (l.source.id === d.id || l.target.id === d.id) ? "#002FA7" : "#E0E0E0")
        .attr("stroke-width", l => (l.source.id === d.id || l.target.id === d.id) ? 3 : 1)
        .attr("stroke-opacity", l => (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.2);
    node.selectAll("circle")
      .attr("opacity", n => {
        if (n.id === d.id) return 1;
        const connected = links.some(l => 
          (l.source.id === d.id && l.target.id === n.id) || 
          (l.target.id === d.id && l.source.id === n.id)
        );
        return connected ? 1 : 0.25;
      });
  })
  .on("mousemove", (event) => {
    tooltip.style("left", (event.pageX + 12) + "px")
           .style("top", (event.pageY - 12) + "px");
  })
  .on("mouseleave", () => {
    tooltip.style("opacity", 0);
    link.attr("stroke", "#CCCCCC")
        .attr("stroke-width", d => Math.max(1, d.weight * 4))
        .attr("stroke-opacity", 0.6);
    node.selectAll("circle").attr("opacity", 1);
  });

  // Touch support for mobile
  node.on("touchstart", (event, d) => {
    event.preventDefault();
    tooltip.style("opacity", 1)
      .html("SRC // " + d.noteTitle + "<br><br>" + d.preview)
      .style("left", "10px").style("top", "10px");
    link.attr("stroke", l => (l.source.id === d.id || l.target.id === d.id) ? "#002FA7" : "#E0E0E0")
        .attr("stroke-width", l => (l.source.id === d.id || l.target.id === d.id) ? 3 : 1);
  });

  // Tick update
  simulation.on("tick", () => {
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    linkLabel.attr("x", d => (d.source.x + d.target.x) / 2)
             .attr("y", d => (d.source.y + d.target.y) / 2);
    node.attr("transform", d => "translate(" + d.x + "," + d.y + ")");
  });

  // Initial zoom to fit
  setTimeout(() => {
    const bounds = g.node().getBBox();
    const dx = bounds.width, dy = bounds.height;
    const x = bounds.x + dx/2, y = bounds.y + dy/2;
    const scale = Math.min(0.9, 0.9 / Math.max(dx / width, dy / height));
    const transform = d3.zoomIdentity.translate(width/2 - scale*x, height/2 - scale*y).scale(scale);
    svg.transition().duration(500).call(zoom.transform, transform);
  }, 1000);
}
</script>
</body>
</html>`;
}

export default function GraphScreen() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [threshold, setThreshold] = useState(0.3);
  const webViewRef = useRef<WebView>(null);

  const fetchGraph = useCallback(async (t: number) => {
    setLoading(true);
    setError('');
    try {
      const resp = await fetch(`${API_URL}/api/graph?threshold=${t}`);
      if (!resp.ok) throw new Error('Failed to load graph');
      const data: GraphData = await resp.json();
      setGraphData(data);
    } catch (err) {
      setError('Could not load graph data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchGraph(threshold); }, []);

  const adjustThreshold = (delta: number) => {
    const newT = Math.max(0.1, Math.min(0.95, Math.round((threshold + delta) * 100) / 100));
    setThreshold(newT);
    fetchGraph(newT);
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header} testID="graph-header">
        <View>
          <Text style={styles.headerTitle}>GRAPH</Text>
          <Text style={styles.headerSub}>
            {graphData ? `${graphData.nodes.length} NODES // ${graphData.edges.length} EDGES` : 'KNOWLEDGE MAP'}
          </Text>
        </View>
        <View style={styles.controls}>
          <TouchableOpacity
            testID="graph-threshold-down"
            style={styles.thresholdBtn}
            onPress={() => adjustThreshold(-0.05)}
            activeOpacity={0.7}
          >
            <Text style={styles.thresholdBtnText}>-</Text>
          </TouchableOpacity>
          <Text style={styles.thresholdLabel} testID="graph-threshold-value">
            {threshold.toFixed(2)}
          </Text>
          <TouchableOpacity
            testID="graph-threshold-up"
            style={styles.thresholdBtn}
            onPress={() => adjustThreshold(0.05)}
            activeOpacity={0.7}
          >
            <Text style={styles.thresholdBtnText}>+</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Info bar showing cached status */}
      {graphData && (
        <View style={styles.infoBar} testID="graph-info-bar">
          <Text style={styles.infoText}>
            THRESHOLD: {graphData.threshold.toFixed(2)} // {graphData.cached ? 'CACHED' : 'COMPUTED'}
          </Text>
        </View>
      )}

      <View style={styles.graphContainer}>
        {loading ? (
          <View style={styles.loadingContainer} testID="graph-loading">
            <Text style={styles.loadingText}>[ COMPUTING GRAPH... ]</Text>
          </View>
        ) : error ? (
          <View style={styles.errorContainer} testID="graph-error">
            <Text style={styles.errorText}>{error}</Text>
            <TouchableOpacity
              testID="graph-retry-btn"
              style={styles.retryBtn}
              onPress={() => fetchGraph(threshold)}
            >
              <Text style={styles.retryBtnText}>[ RETRY ]</Text>
            </TouchableOpacity>
          </View>
        ) : graphData && graphData.nodes.length > 0 ? (
          <WebView
            ref={webViewRef}
            testID="graph-webview"
            source={{ html: buildGraphHtml(graphData) }}
            style={styles.webview}
            javaScriptEnabled={true}
            domStorageEnabled={true}
            scrollEnabled={false}
            bounces={false}
            originWhitelist={['*']}
          />
        ) : (
          <View style={styles.emptyContainer} testID="graph-empty-state">
            <Text style={styles.emptyTitle}>NO GRAPH</Text>
            <Text style={styles.emptyHint}>
              Add notes to build a knowledge graph.{'\n'}
              Edges appear when chunks are semantically similar.
            </Text>
          </View>
        )}
      </View>

      {/* Legend */}
      <View style={styles.legend} testID="graph-legend">
        <Text style={styles.legendText}>
          PINCH TO ZOOM // DRAG NODES // TAP TO HIGHLIGHT
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#FFFFFF' },
  header: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: 16, paddingTop: 16, paddingBottom: 12,
    borderBottomWidth: 2, borderBottomColor: '#0A0A0A',
  },
  headerTitle: {
    fontFamily: 'Courier', fontWeight: '900', fontSize: 28,
    color: '#0A0A0A', letterSpacing: -1,
  },
  headerSub: {
    fontFamily: 'Courier', fontSize: 11, color: '#555555',
    letterSpacing: 2, marginTop: 2,
  },
  controls: {
    flexDirection: 'row', alignItems: 'center',
  },
  thresholdBtn: {
    width: 36, height: 36,
    borderWidth: 2, borderColor: '#0A0A0A',
    justifyContent: 'center', alignItems: 'center',
  },
  thresholdBtnText: {
    fontFamily: 'Courier', fontWeight: '900', fontSize: 18,
    color: '#0A0A0A',
  },
  thresholdLabel: {
    fontFamily: 'Courier', fontWeight: '700', fontSize: 14,
    color: '#002FA7', marginHorizontal: 10, letterSpacing: 1,
  },
  infoBar: {
    backgroundColor: '#F5F5F5',
    borderBottomWidth: 1, borderBottomColor: '#E0E0E0',
    paddingVertical: 6, paddingHorizontal: 16,
  },
  infoText: {
    fontFamily: 'Courier', fontSize: 10, color: '#555555',
    letterSpacing: 2,
  },
  graphContainer: { flex: 1 },
  webview: { flex: 1, backgroundColor: '#FFFFFF' },
  loadingContainer: {
    flex: 1, justifyContent: 'center', alignItems: 'center',
  },
  loadingText: {
    fontFamily: 'Courier', fontWeight: '600', fontSize: 14,
    color: '#0A0A0A', letterSpacing: 2,
  },
  errorContainer: {
    flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 32,
  },
  errorText: {
    fontFamily: 'Courier', fontSize: 14, color: '#FF2A00',
    textAlign: 'center', letterSpacing: 1,
  },
  retryBtn: {
    marginTop: 16, borderWidth: 1, borderColor: '#0A0A0A',
    paddingHorizontal: 20, paddingVertical: 10,
  },
  retryBtnText: {
    fontFamily: 'Courier', fontWeight: '600', fontSize: 12,
    color: '#0A0A0A', letterSpacing: 1,
  },
  emptyContainer: {
    flex: 1, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 32,
  },
  emptyTitle: {
    fontFamily: 'Courier', fontWeight: '900', fontSize: 24,
    color: '#0A0A0A',
  },
  emptyHint: {
    fontFamily: 'Courier', fontSize: 13, color: '#555555',
    textAlign: 'center', marginTop: 12, lineHeight: 20,
  },
  legend: {
    borderTopWidth: 2, borderTopColor: '#0A0A0A',
    paddingVertical: 8, paddingHorizontal: 16,
    backgroundColor: '#F5F5F5',
  },
  legendText: {
    fontFamily: 'Courier', fontSize: 10, color: '#555555',
    letterSpacing: 2, textAlign: 'center',
  },
});
