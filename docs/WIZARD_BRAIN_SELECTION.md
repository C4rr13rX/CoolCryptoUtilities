# Wizard brain selection

CoolCryptoUtilities keeps three Wizard-brain choices independent:

1. **Operations brain** — selected in Model Control and used by ordinary
   C0D3R V2 web runs.
2. **Wizard Chat brain** — selected in Wizard Chat and used by chat,
   live training, pool/statistics views, and Wizard Chat agent mode.
3. **BrandDozer run brain** — selected when a delivery or research run is
   created. BrandDozer remains the Scrum/research orchestrator, C0D3R V2
   remains the agent, and the selected Wizard brain is C0D3R V2's model.

## Registry

Named brains are maintained in Model Control. A profile contains:

- a user-owned UUID and display name;
- an absolute HTTP(S) node base URL;
- `/brain/chat` for a merged Wizard node or `/chat` for a standalone
  brain server.

The environment-default profile is derived from
`WIZARD_BRAIN_CHAT_URL`, `WIZARD_BRAIN_URL`, or `WIZARD_NODE_URL` and
cannot be deleted. Additional profiles live in the user's encrypted-vault
database namespace as non-secret configuration.

## Durable routing

A queued C0D3R web run persists `wizard_brain_id`, `wizard_endpoint`, and
`wizard_chat_path`. A BrandDozer run stores the same values in its durable
context. Workers use these snapshots instead of looking up the current
dropdown later. Therefore:

- changing Model Control does not redirect queued or running work;
- changing Wizard Chat does not affect operations or BrandDozer;
- a process-wide default such as AgentTheFreeloader cannot override an
  explicit Wizard selection made for a BrandDozer run.

Cached C0D3R flows are rebuilt when the selected route changes.
