(() => {
  'use strict';

  const scope = typeof window === 'undefined' ? globalThis : window;
  const { createSvgElement: element } = scope.StateNetworkUI || {};
  const PALETTE = [
    '#66d9c2', '#ff9c73', '#72adff', '#c6a7ff',
    '#e9ca68', '#f47f9b', '#89ce79', '#b8c4d8',
  ];

  class HistoryBuffer {
    constructor(capacity = 360) {
      this.capacity = capacity;
      this.values = new Array(capacity);
      this.start = 0;
      this.length = 0;
    }

    clear() {
      this.start = 0;
      this.length = 0;
    }

    push(value) {
      if (this.length < this.capacity) {
        this.values[(this.start + this.length) % this.capacity] = value;
        this.length += 1;
        return;
      }
      this.values[this.start] = value;
      this.start = (this.start + 1) % this.capacity;
    }

    toArray(limit = this.length) {
      const count = Math.min(this.length, limit);
      const offset = this.length - count;
      return Array.from(
        { length: count },
        (_, index) => this.values[(this.start + offset + index) % this.capacity],
      );
    }
  }

  function stateColor(value, scale) {
    const amount = Math.min(1, Math.abs(value) / scale);
    const base = value >= 0 ? [255, 145, 107] : [96, 171, 255];
    const mixed = base.map((component) => Math.round(202 + (component - 202) * amount));
    return `rgb(${mixed.join(',')})`;
  }

  function sameValues(left, right) {
    return left.length === right.length && left.every((value, index) => value === right[index]);
  }

  function captureTopology(network) {
    return {
      nodes: Number(network.nodes),
      inputs: [...network.input_nodes],
      actions: [...network.action_nodes],
      edges: network.edges.map((edge) => [edge.source, edge.target]),
    };
  }

  function sameTopology(network, topology) {
    return topology !== null
      && Number(network.nodes) === topology.nodes
      && sameValues(network.input_nodes, topology.inputs)
      && sameValues(network.action_nodes, topology.actions)
      && network.edges.length === topology.edges.length
      && network.edges.every((edge, index) => (
        edge.source === topology.edges[index][0] && edge.target === topology.edges[index][1]
      ));
  }

  function networkPositions(network, width, height) {
    const inputs = network.input_nodes;
    const actions = network.action_nodes;
    const boundaries = new Set([...inputs, ...actions]);
    const hidden = Array.from(
      { length: Number(network.nodes) },
      (_, index) => index,
    ).filter((index) => !boundaries.has(index));
    const positions = new Array(Number(network.nodes));
    const spread = (nodes, x, top, bottom) => nodes.forEach((node, index) => {
      positions[node] = {
        x,
        y: nodes.length === 1
          ? (top + bottom) / 2
          : top + index * (bottom - top) / Math.max(1, nodes.length - 1),
      };
    });
    spread(inputs, 132, 62, height - 44);
    spread(actions, width - 132, 190, height - 190);
    const columns = Math.max(1, Math.ceil(Math.sqrt(hidden.length * 1.25)));
    const rows = Math.max(1, Math.ceil(hidden.length / columns));
    hidden.forEach((node, index) => {
      positions[node] = {
        x: 300 + (index % columns) * 320 / Math.max(1, columns - 1),
        y: 74 + Math.floor(index / columns) * (height - 148) / Math.max(1, rows - 1),
      };
    });
    return positions;
  }

  class NetworkView {
    constructor(svg, { width = 920, height = 560 } = {}) {
      this.svg = svg;
      this.width = width;
      this.height = height;
      this.topology = null;
      this.edgeElements = [];
      this.nodeElements = [];
    }

    reset() {
      this.topology = null;
      this.edgeElements = [];
      this.nodeElements = [];
      this.svg.replaceChildren();
    }

    render(network, { channel = 0, edgeThreshold = 0, inputLabels = [] } = {}) {
      if (!sameTopology(network, this.topology)) {
        this.build(network, inputLabels);
        this.topology = captureTopology(network);
      }
      this.update(network, channel, edgeThreshold);
    }

    build(network, inputLabels) {
      const positions = networkPositions(network, this.width, this.height);
      this.svg.replaceChildren();
      this.edgeElements = [];
      this.nodeElements = [];

      const defs = element('defs');
      const marker = element('marker', {
        id: 'demo-arrow', markerWidth: '7', markerHeight: '7',
        refX: '6', refY: '3.5', orient: 'auto',
      });
      marker.append(element('path', { d: 'M0,0 L7,3.5 L0,7 z', fill: '#71809b' }));
      defs.append(marker);

      const zones = element('g', { class: 'demo-network-zones' });
      zones.append(
        element('rect', { x: 0, y: 0, width: 252, height: this.height, class: 'demo-network-zone' }),
        element('rect', { x: this.width - 252, y: 0, width: 252, height: this.height, class: 'demo-network-zone' }),
      );
      const edgeLayer = element('g', { class: 'demo-network-edges' });
      network.edges.forEach((edge, index) => {
        const from = positions[edge.source];
        const to = positions[edge.target];
        const line = element('line', {
          x1: from.x, y1: from.y, x2: to.x, y2: to.y,
          'marker-end': 'url(#demo-arrow)', class: 'edge',
        });
        const title = element('title');
        title.textContent = `edge ${index}: ${edge.source} → ${edge.target}`;
        line.append(title);
        edgeLayer.append(line);
        this.edgeElements.push(line);
      });

      const nodeLayer = element('g', { class: 'demo-network-nodes' });
      positions.forEach((point, index) => {
        const group = element('g');
        const circle = element('circle', {
          cx: point.x, cy: point.y, r: '11', class: 'node',
        });
        const label = element('text', {
          x: point.x, y: point.y + 4, class: 'node-label',
        });
        const title = element('title');
        title.textContent = `node ${index}`;
        circle.append(title);
        label.textContent = index;
        group.append(circle, label);
        nodeLayer.append(group);
        this.nodeElements.push(circle);
      });

      const annotationLayer = element('g', { class: 'demo-network-annotations' });
      const heading = (text, x, anchor) => {
        const item = element('text', {
          x, y: 26, class: 'demo-network-heading', 'text-anchor': anchor,
        });
        item.textContent = text;
        annotationLayer.append(item);
      };
      heading('SENSORY PORTS', 18, 'start');
      heading('ACTION PORTS', this.width - 18, 'end');
      network.input_nodes.forEach((node, index) => {
        const point = positions[node];
        const text = element('text', {
          x: 18, y: point.y + 4, class: 'demo-boundary-label',
          'text-anchor': 'start',
        });
        const signalChannel = network.input_signal_channels?.[index] ?? 0;
        text.textContent = `${inputLabels[index] || `input ${index}`} · c${signalChannel}`;
        annotationLayer.append(text);
      });
      const actionLabels = ['turn', 'throttle'];
      network.action_nodes.forEach((node, index) => {
        const point = positions[node];
        const text = element('text', {
          x: this.width - 18, y: point.y + 4, class: 'demo-boundary-label',
          'text-anchor': 'end',
        });
        text.textContent = actionLabels[index] || `action ${index}`;
        annotationLayer.append(text);
      });
      this.svg.append(defs, zones, edgeLayer, nodeLayer, annotationLayer);
    }

    update(network, channel, edgeThreshold) {
      let scale = 0.15;
      for (const vector of network.node_state) {
        scale = Math.max(scale, Math.abs(Number(vector[channel] || 0)));
      }
      this.nodeElements.forEach((circle, index) => {
        const value = Number(network.node_state[index]?.[channel] || 0);
        circle.setAttribute('fill', stateColor(value, scale));
        circle.firstElementChild.textContent = `node ${index} · channel ${channel}: ${value.toFixed(5)}`;
      });
      this.edgeElements.forEach((line, index) => {
        const strength = Number(network.edges[index]?.communication_strength ?? 1);
        const visible = Math.abs(strength) >= edgeThreshold;
        line.style.display = visible ? '' : 'none';
        line.setAttribute('stroke', strength >= 0 ? '#ff9c73' : '#72adff');
        line.setAttribute('stroke-width', String(0.65 + Math.min(2.8, Math.abs(strength) * 2.1)));
        line.setAttribute('opacity', String(0.08 + Math.min(0.68, Math.abs(strength) * 0.72)));
      });
    }
  }

  function drawNodeHistoryChart(
    canvas,
    entries,
    { channel = 0, seriesCount = 8 } = {},
  ) {
    const context = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#0b1119';
    context.fillRect(0, 0, width, height);
    if (!entries.length) {
      context.fillStyle = '#8391a5';
      context.font = '14px system-ui';
      context.fillText('Select an organism to collect state history.', 22, 34);
      return [];
    }

    const nodeCount = entries.at(-1).node_state.length;
    const energy = new Array(nodeCount).fill(0);
    let scale = 0.1;
    for (const entry of entries) {
      for (let node = 0; node < nodeCount; node += 1) {
        const value = Number(entry.node_state[node]?.[channel] || 0);
        energy[node] += value * value;
        scale = Math.max(scale, Math.abs(value));
      }
    }
    scale *= 1.12;
    const rankedNodes = Array.from({ length: nodeCount }, (_, node) => node)
      .sort((left, right) => energy[right] - energy[left]);
    const nodes = seriesCount > 0 ? rankedNodes.slice(0, seriesCount) : rankedNodes;
    const left = 54;
    const right = width - 18;
    const top = 20;
    const bottom = height - 40;
    const x = (index) => left + index / Math.max(1, entries.length - 1) * (right - left);
    const y = (value) => (top + bottom) / 2 - value / scale * (bottom - top) / 2;

    context.font = '11px ui-monospace';
    context.lineWidth = 1;
    for (let grid = -2; grid <= 2; grid += 1) {
      const value = scale * grid / 2;
      const py = y(value);
      context.strokeStyle = grid === 0 ? '#526174' : '#243142';
      context.beginPath();
      context.moveTo(left, py);
      context.lineTo(right, py);
      context.stroke();
      context.fillStyle = '#718096';
      context.fillText(value.toFixed(2), 6, py + 4);
    }

    nodes.forEach((node, rank) => {
      context.strokeStyle = PALETTE[rank % PALETTE.length];
      context.globalAlpha = rank < 4 ? 0.96 : 0.7;
      context.lineWidth = rank < 3 ? 2 : 1.25;
      context.beginPath();
      entries.forEach((entry, index) => {
        const value = Number(entry.node_state[node]?.[channel] || 0);
        if (index === 0) context.moveTo(x(index), y(value));
        else context.lineTo(x(index), y(value));
      });
      context.stroke();
    });
    context.globalAlpha = 1;
    context.fillStyle = '#8391a5';
    const firstTick = entries[0].tick;
    const lastTick = entries.at(-1).tick;
    context.fillText(`tick ${firstTick}`, left, height - 14);
    context.textAlign = 'right';
    context.fillText(`tick ${lastTick}`, right, height - 14);
    context.textAlign = 'left';

    return nodes.map((node, rank) => ({
      node,
      color: PALETTE[rank % PALETTE.length],
      rms: Math.sqrt(energy[node] / entries.length),
      current: Number(entries.at(-1).node_state[node]?.[channel] || 0),
    }));
  }

  function nearestOrganism(world, worldX, worldY) {
    let nearest = null;
    let distance = Number.POSITIVE_INFINITY;
    for (const organism of world.organisms) {
      const candidate = Math.hypot(worldX - organism.x, worldY - organism.y);
      if (candidate < distance) {
        nearest = organism;
        distance = candidate;
      }
    }
    return nearest ? { organism: nearest, distance } : null;
  }

  scope.StateNetworkEcology = Object.freeze({
    HistoryBuffer,
    NetworkView,
    drawNodeHistoryChart,
    nearestOrganism,
  });
})();
