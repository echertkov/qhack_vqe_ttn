# Ansätze
---
We use the following ansätze in our work

## Multi-Scale Entanglement Renormalization Ansatz (MERA)
```
  0: ──╭C──╭C──────╭C──────╭C──────┤
  1: ──│───│───────│───────╰X──╭C──┤
  2: ──│───│───────╰X──╭C──╭C──╰X──┤
  3: ──│───│───────────│───╰X──╭C──┤
  4: ──│───╰X──╭C──╭C──╰X──╭C──╰X──┤
  5: ──│───────│───│───────╰X──╭C──┤
  6: ──│───────│───╰X──╭C──╭C──╰X──┤
  7: ──│───────│───────│───╰X──╭C──┤
  8: ──╰X──╭C──╰X──╭C──╰X──╭C──╰X──┤
  9: ──────│───────│───────╰X──╭C──┤
 10: ──────│───────╰X──╭C──╭C──╰X──┤
 11: ──────│───────────│───╰X──╭C──┤
 12: ──────╰X──────╭C──╰X──╭C──╰X──┤
 13: ──────────────│───────╰X──╭C──┤
 14: ──────────────╰X──────╭C──╰X──┤
 15: ──────────────────────╰X──────┤
```
where the CNOTs represent arbitrary unitary parameterizations (disentanglers or isometries)

## Tree Tensor Network
```
  0: ──╭C──╭C──╭C──╭C──┤ 
  1: ──│───│───│───╰X──┤     
  2: ──│───│───╰X──╭C──┤     
  3: ──│───│───────╰X──┤     
  4: ──│───╰X──╭C──╭C──┤     
  5: ──│───────│───╰X──┤     
  6: ──│───────╰X──╭C──┤     
  7: ──│───────────╰X──┤     
  8: ──╰X──╭C──╭C──╭C──┤     
  9: ──────│───│───╰X──┤     
 10: ──────│───╰X──╭C──┤     
 11: ──────│───────╰X──┤     
 12: ──────╰X──╭C──╭C──┤     
 13: ──────────│───╰X──┤     
 14: ──────────╰X──╭C──┤     
 15: ──────────────╰X──┤     


```

## Hardware Efficient Ansatz
Our Ansatz is based off of the BasicEntanglerLayer template in pennylane:
https://pennylane.readthedocs.io/en/stable/code/api/pennylane.templates.layers.BasicEntanglerLayers.html

