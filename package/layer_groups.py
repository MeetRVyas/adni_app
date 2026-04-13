def get_swin_groups(model):
    """Swin Transformer layer groups."""
    # We initialize 5 groups as per your original logic
    groups = [[] for _ in range(5)]
    
    # We use a set to track parameter IDs to ensure 100% coverage
    param_ids = set()
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # --- Logic matching your original context ---
        
        # Group 4: Head
        # timm swin models name the classifier "head"
        if name.startswith('head.'):
            groups[4].append(param)
            
        # Group 0: Patch Embed (Stem)
        elif name.startswith('patch_embed.') or name.startswith('absolute_pos_embed'):
            groups[0].append(param)
            
        # Layers (Stages 1-4)
        elif name.startswith('layers.'):
            # name format is "layers.X.blocks..."
            # We parse X to determine the group
            try:
                # Extract the layer index (0, 1, 2, or 3)
                layer_idx = int(name.split('.')[1])
                
                if layer_idx == 0:
                    # Context: Group 0 includes Stage 1
                    groups[0].append(param)
                elif layer_idx == 1:
                    # Context: Group 1 is Stage 2
                    groups[1].append(param)
                elif layer_idx == 2:
                    # Context: Group 2 is Stage 3
                    groups[2].append(param)
                elif layer_idx == 3:
                    # Context: Group 3 is Stage 4
                    groups[3].append(param)
                else:
                    # Fallback: If model has >4 stages (rare), put in Group 3
                    groups[3].append(param)
                    
            except (IndexError, ValueError):
                # Fallback: If parsing fails, put in Group 0 (safest default)
                print(f"Warning: Could not parse layer index for {name}. Assigning to Group 0.")
                groups[0].append(param)

        # Group 3: Final Norm
        # Context: Your code put model.norm in Group 3
        elif name.startswith('norm.'):
            groups[3].append(param)
            
        # Catch-all
        else:
            print(f"Warning: Unknown parameter found: '{name}'. Assigning to Group 0.")
            groups[0].append(param)
        
        # Track ID
        param_ids.add(id(param))

    # --- Robustness Check ---
    # Ensure every single trainable parameter was assigned to a group
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(param_ids) != len(trainable_params):
        raise RuntimeError(
            f"Grouping Error: Model has {len(trainable_params)} trainable params, "
            f"but function only grouped {len(param_ids)}. "
            "Check for frozen layers or shared parameters."
    )

    for i, group in enumerate(groups) :
        print(f"Group {i} -> {len(group)}")

    return groups