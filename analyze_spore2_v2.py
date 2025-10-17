import json

with open('spores/v16_picker/scripts/run/buffer/real_graph_latest.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Спора 2 (index=1, spore_id="3")
spore = data['spores'][1]
print(f"=== Spore 2 (internal ID: {spore['spore_id']}) ===\n")

print('OUTGOING LINKS (children):')
for link in spore['out_links']:
    print(f"  Link {link['link_number']}: to spore {link['to_spore_id']}")
    print(f"    control={link['control']}, dt={link['dt']}, dt_sign={link['dt_sign']}, raw_dt={link.get('raw_dt', 'N/A')}")
    direction = 'forward' if link['dt_sign'] == 1 else 'backward'
    control_type = 'max' if link['control'] > 0 else 'min'
    print(f"    => {direction}_{control_type}\n")

print('INCOMING LINKS (going backwards along the edge):')
for link in spore['in_links']:
    print(f"  Link {link['link_number']}: from spore {link['from_spore_id']}")
    print(f"    Edge: control={link['control']}, dt={link['dt']}, dt_sign={link['dt_sign']}")

    # When going backwards:
    # - dt stays the same (we use the raw value from edge)
    # - control gets inverted
    # - time direction: if edge was forward (dt_sign=1), backwards means we use -dt
    #                   if edge was backward (dt_sign=-1), backwards means we use +dt

    # But actually, in the edge: raw_dt = dt * dt_sign
    # So if dt_sign=-1, raw_dt is already negative

    # When going backwards, we invert the control
    inverted_control = -link['control']
    control_type = 'max' if inverted_control > 0 else 'min'

    # For time direction: if dt_sign=1 (forward edge), going back is backward
    #                      if dt_sign=-1 (backward edge), going back is forward
    if link['dt_sign'] == 1:
        time_direction = 'backward'
        final_dt = link['dt']  # positive dt, but we call it backward because we're going back in time
    else:  # dt_sign == -1
        time_direction = 'backward'  # Hmm this doesn't seem right
        final_dt = link['dt']

    print(f"    Going backwards: control={inverted_control} ({control_type})")
    print(f"    Time direction analysis:")
    print(f"      - Edge dt_sign={link['dt_sign']} means edge goes {'forward' if link['dt_sign']==1 else 'backward'}")
    print(f"      - Going against a {'forward' if link['dt_sign']==1 else 'backward'} edge...")
    print()

print("\n=== What we expect (from user description) ===")
print("forward_max → 3")
print("forward_min → 5")
print("backward_max → 6")
print("backward_min → 4")
