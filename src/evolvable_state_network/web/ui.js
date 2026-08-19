(() => {
  'use strict';

  const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';

  function byId(id) {
    return document.getElementById(id);
  }

  function createSvgElement(tag, attributes = {}) {
    const node = document.createElementNS(SVG_NAMESPACE, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function formatNumber(value, digits = 5) {
    return Number.isFinite(value) ? Number(value).toFixed(digits) : '—';
  }

  function apiError(data, fallback) {
    if (Array.isArray(data?.detail)) {
      return data.detail.map((item) => item.msg).join('; ');
    }
    return data?.detail || data?.error || fallback;
  }

  function statCard(label, value) {
    const card = document.createElement('article');
    card.className = 'stat';
    const title = document.createElement('span');
    title.textContent = label;
    const result = document.createElement('strong');
    result.textContent = value;
    card.append(title, result);
    return card;
  }

  function definitionRows(rows) {
    return rows.flatMap(([term, value]) => {
      const name = document.createElement('dt');
      const description = document.createElement('dd');
      name.textContent = term;
      description.textContent = value;
      return [name, description];
    });
  }

  window.StateNetworkUI = Object.freeze({
    byId,
    createSvgElement,
    formatNumber,
    apiError,
    statCard,
    definitionRows,
  });
})();
