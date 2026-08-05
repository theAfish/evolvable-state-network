# Frontend guide

The lab UI is deliberately build-free: FastAPI serves this directory directly,
so a Python server restart or `--reload` is enough during development.

## File boundaries

- `index.html` owns semantic structure and stable element IDs used by the app.
- `app.js` owns API requests, simulation state, rendering, and view routing.
- `style.css` owns simulation-specific components and visualizations.
- `shell.css` owns design tokens, navigation, common forms, accessibility, and
  responsive layout. New shared UI rules should normally start here.

Keeping shell rules separate prevents page-level work from adding another layer
of overrides to the original simulation stylesheet.

## Run locally

From the repository root:

```powershell
evolvable-state-network-server --reload
```

Open `http://127.0.0.1:8000/`. Each workspace has a stable hash URL, such as
`/#survival`, `/#embodied`, or `/#replay`, which is useful when reproducing UI
issues. Form choices are retained in browser storage; clear site data to return
all controls to repository defaults.

## Change safely

Element IDs referenced by the `ui` map at the top of `app.js` form the internal
DOM contract. When renaming one, update the HTML and map together. API payloads
should continue to be assembled close to the action that submits them.

Before handing off a frontend change, run:

```powershell
python -m unittest tests.test_server
```

Then check keyboard focus, narrow-screen layout, the six hash routes, and at
least one server-backed workflow. The root-page server test intentionally checks
key survival-training language as a lightweight smoke test.
