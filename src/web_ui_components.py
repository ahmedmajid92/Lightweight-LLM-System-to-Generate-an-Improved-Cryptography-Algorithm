# src/web_ui_components.py
import gradio as gr
from typing import Dict, List
from .component_registry import list_algorithms, list_components_for_algorithm, scan_components
from .algorithm_blueprints import get_blueprint, list_blueprints
from .schemas import CompositionRequest
from .composer import compose
from .validator import validate_composed
from .chat_orchestrator import ChatOrchestrator

def _components_table_data():
    """Get all components from Components.py as table rows."""
    alg_map = scan_components()
    rows = []
    for alg, comps in alg_map.items():
        for c in comps:
            rows.append([
                False,  # Selection checkbox
                alg, 
                c.stage or "-", 
                c.name, 
                c.signature_str, 
                (c.doc[:120] + "…") if len(c.doc) > 120 else c.doc
            ])
    return rows

def _get_all_component_names():
    """Get a list of all component function names."""
    alg_map = scan_components()
    names = []
    for comps in alg_map.values():
        for c in comps:
            names.append(c.name)
    return names

def build_ui():
    chat = ChatOrchestrator()

    with gr.Blocks(title="Crypto Builder + Chat", css="footer {visibility:hidden}") as demo:
        gr.Markdown("## 🔐 Cryptography Builder — compose components, validate, and chat")
        gr.Markdown("**Step 1:** Browse and select components → **Step 2:** Compose your algorithm → **Step 3:** Chat for help")
        
        # Shared state to store selected components
        selected_components = gr.State([])

        with gr.Tabs():
            with gr.Tab("📚 1. Browse & Select Components"):
                gr.Markdown("### Select the cryptographic components you want to use")
                gr.Markdown("Browse all available components and select the ones you need")
                gr.Markdown("💡 **Tip:** Components are organized by stage type - all Key Schedule functions are grouped together, all Sub Bytes together, etc.")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Reference table (read-only) to show all components
                        gr.Markdown("**All Available Components:**")
                        reference_table = gr.Dataframe(
                        headers=["Algorithm", "Stage", "Function", "Signature", "Doc"],
                            value=[[row[1], row[2], row[3], row[4], row[5]] for row in _components_table_data()],
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Selection Panel")
                        gr.Markdown("Check the components you want to use:")
                        
                        # Create component choices for CheckboxGroup, organized by stage
                        def get_component_choices():
                            alg_map = scan_components()
                            
                            # Group by stage first
                            stage_groups = {}
                            for alg, comps in alg_map.items():
                                for c in comps:
                                    stage = c.stage or "other"
                                    if stage not in stage_groups:
                                        stage_groups[stage] = []
                                    # Format: "[STAGE] Algorithm | Function"
                                    stage_label = stage.replace('_', ' ').title()
                                    label = f"[{stage_label}] {alg} | {c.name}"
                                    stage_groups[stage].append((label, alg, c.name, stage))
                            
                            # Build ordered list with stage groups
                            choices = []
                            # Primary stages (common in algorithms)
                            priority_stages = ["key_schedule", "sub_bytes", "shift_rows", "mix_columns", 
                                             "add_round_key", "round_function", "permute"]
                            
                            # Add components from priority stages first
                            for stage in priority_stages:
                                if stage in stage_groups:
                                    stage_groups[stage].sort(key=lambda x: (x[1], x[2]))
                                    choices.extend([item[0] for item in stage_groups[stage]])
                            
                            # Add ALL remaining stages (so nothing is hidden!)
                            remaining_stages = sorted(set(stage_groups.keys()) - set(priority_stages))
                            for stage in remaining_stages:
                                stage_groups[stage].sort(key=lambda x: (x[1], x[2]))
                                choices.extend([item[0] for item in stage_groups[stage]])
                            
                            return choices
                        
                        component_selector = gr.CheckboxGroup(
                            choices=get_component_choices(),
                            label="Select Components",
                            value=[],
                            interactive=True
                        )
                        
                        with gr.Row():
                            select_all_btn = gr.Button("✓ Select All", size="sm")
                            clear_all_btn = gr.Button("✗ Clear All", size="sm")
                        
                        selected_count = gr.Markdown("**0** components selected")
                
                save_selection_btn = gr.Button("💾 Save Selection for Compose", variant="primary", size="lg")
                selection_status = gr.Markdown("")
                
                def select_all_components():
                    """Select all components."""
                    return get_component_choices()
                
                def clear_all_components():
                    """Clear all selections."""
                    return []
                
                def update_count(selected_items):
                    """Update the count of selected components."""
                    count = len(selected_items) if selected_items else 0
                    return f"**{count}** components selected"
                
                def save_selected_from_checkboxes(selected_items):
                    """Convert selected checkbox items back to component details."""
                    if not selected_items:
                        return [], "⚠️ No components selected! Please check some boxes first."
                    
                    # Get full component data
                    alg_map = scan_components()
                    selected = []
                    
                    for item in selected_items:
                        # Parse: "[STAGE] Algorithm | Function"
                        # Extract stage from brackets
                        stage_match = item.split(']', 1)
                        if len(stage_match) >= 2:
                            stage_part = stage_match[0].replace('[', '').strip()
                            rest = stage_match[1].strip()
                            
                            # Parse algorithm and function
                            parts = rest.split(" | ")
                            if len(parts) >= 2:
                                alg = parts[0].strip()
                                name = parts[1].strip()
                                
                                # Convert stage label back to stage name
                                stage = stage_part.lower().replace(' ', '_')
                                
                                # Find the component details
                                if alg in alg_map:
                                    for c in alg_map[alg]:
                                        if c.name == name:
                                            selected.append({
                                                "algorithm": alg,
                                                "stage": stage if stage != "other" else "",
                                                "name": name,
                                                "signature": c.signature_str,
                                                "doc": c.doc
                                            })
                                            break
                    
                    return selected, f"✅ Saved {len(selected)} components! Go to 'Compose Algorithm' tab to build your cipher."
                
                # Wire up the buttons
                select_all_btn.click(fn=select_all_components, outputs=[component_selector])
                clear_all_btn.click(fn=clear_all_components, outputs=[component_selector])
                component_selector.change(fn=update_count, inputs=[component_selector], outputs=[selected_count])
                save_selection_btn.click(
                    fn=save_selected_from_checkboxes, 
                    inputs=[component_selector], 
                    outputs=[selected_components, selection_status]
                )
            with gr.Tab("🔧 2. Compose Algorithm") as compose_tab:
                gr.Markdown("### Build your custom cipher from selected components")
                gr.Markdown("Choose which component to use for each stage of the algorithm")
                
                with gr.Row():
                    blueprint_drop = gr.Dropdown(
                        choices=list_blueprints(), 
                        value="AES", 
                        label="Base Algorithm Structure"
                    )
                    out_name = gr.Textbox(
                        label="Output Algorithm Name", 
                        value="hybrid_cipher"
                    )
                
                stage_status = gr.Markdown("ℹ️ Select components in Tab 1 first, then choose an algorithm above")
                
                # Create stage selectors for all possible stages
                with gr.Column():
                    all_stages = ["key_schedule", "sub_bytes", "shift_rows", "mix_columns", 
                                  "add_round_key", "round_function", "permute"]
                    stage_selectors: Dict[str, gr.Dropdown] = {}
                    
                    for stage_name in all_stages:
                        stage_selectors[stage_name] = gr.Dropdown(
                            choices=[], 
                            label=stage_name.replace('_', ' ').title(), 
                            visible=False
                        )
                
                refresh_btn = gr.Button("🔄 Refresh Components", variant="secondary")
                generate_btn = gr.Button("⚡ Generate Algorithm", variant="primary")
                validate_btn = gr.Button("✅ Validate Generated Algorithm", variant="secondary")
                code_preview = gr.Code(label="Generated Algorithm Code", language="python")
                validate_md = gr.Markdown()

                def load_selected_for_compose(base_alg: str, selected_comps):
                    """Load selected components into stage dropdowns based on base algorithm."""
                    if not selected_comps:
                        return [gr.update(visible=False, choices=[], value=None) for _ in all_stages] + [
                            "⚠️ No components selected! Go to 'Browse & Select Components' tab first."
                        ]
                    
                    bp = get_blueprint(base_alg)
                    
                    # Organize selected components by stage
                    choices_map = {s.name: [] for s in bp.stages}
                    for comp in selected_comps:
                        stage = comp["stage"]
                        if stage != "-" and stage in choices_map:
                            choices_map[stage].append(comp["name"])
                    
                    # Sort for consistency
                    for stage in choices_map:
                        choices_map[stage].sort()
                    
                    # Build updates for each stage selector
                    updates = []
                    active_stages = [s.name for s in bp.stages]
                    
                    for stage_name in all_stages:
                        if stage_name in active_stages and stage_name in choices_map:
                            choices = choices_map[stage_name]
                            if choices:
                                updates.append(gr.update(
                                    choices=choices,
                                    value=choices[0],
                                    visible=True,
                                    label=f"{stage_name.replace('_', ' ').title()} ({len(choices)} available)"
                                ))
                            else:
                                updates.append(gr.update(
                                    visible=True,
                                    choices=[],
                                    value=None,
                                    label=f"{stage_name.replace('_', ' ').title()} (⚠️ no components selected)"
                                ))
                        else:
                            updates.append(gr.update(visible=False, choices=[], value=None))
                    
                    spec = f"**{bp.name}** • Structure: {bp.structure}, Rounds: {bp.rounds}, Block: {bp.block_size} bytes, Key: {bp.key_size} bytes"
                    total_comps = sum(len(choices_map[s]) for s in choices_map)
                    status = f"{spec}\n\n✅ Loaded {total_comps} components for {len(active_stages)} stages"
                    
                    return updates + [status]

                def do_generate(base_alg: str, output_name: str, *stage_values):
                    """Generate the composed algorithm."""
                    bp = get_blueprint(base_alg)
                    
                    # Map stage values to stage names
                    selections = {}
                    for i, stage_name in enumerate(all_stages):
                        if i < len(stage_values) and stage_values[i]:
                            selections[stage_name] = stage_values[i]
                    
                    if not selections:
                        return "", "❌ No components selected for any stage! Please select components first."
                    
                    req = CompositionRequest(
                        base_algorithm=base_alg, 
                        selections=selections, 
                        output_name=output_name
                    )
                    result = compose(bp, req)
                    
                    if not result.ok:
                        return "", f"❌ Generation failed:\n\n" + "\n".join(f"- {e}" for e in result.errors)
                    
                    return result.module_code, f"✅ **Generated successfully!**\n\nSaved to: `{result.module_path}`"

                def do_validate(base_alg: str, output_name: str, *stage_values):
                    """Validate the generated algorithm with detailed, user-friendly feedback."""
                    bp = get_blueprint(base_alg)
                    path = f"data/generated_algorithms/{output_name}.py"
                    rep = validate_composed(path, bp)
                    
                    # Build user-friendly validation report
                    if rep.ok:
                        head = "## ✅ Validation Passed!"
                        body = "\n\n🎉 **Congratulations!** Your algorithm is working correctly.\n\n"
                        
                        # Show test results
                        if rep.details and "test_results" in rep.details:
                            body += "**Test Results:**\n"
                            for result in rep.details["test_results"]:
                                body += f"{result}\n"
                        
                        body += f"\n**Algorithm Details:**\n"
                        body += f"- Block Size: {rep.details.get('block_size', 'N/A')} bytes\n"
                        body += f"- Key Size: {rep.details.get('key_size', 'N/A')} bytes\n"
                        body += f"- Encryption Signature: `{rep.details.get('enc_sig', 'N/A')}`\n"
                        body += f"- Decryption Signature: `{rep.details.get('dec_sig', 'N/A')}`\n"
                        
                        if rep.warnings:
                            body += "\n\n### ⚠️ Warnings (non-critical):\n"
                            for warning in rep.warnings:
                                parsed = _parse_validation_message(warning)
                                body += f"\n**{parsed['title']}**\n"
                                body += f"_{parsed['description']}_\n\n"
                                if parsed['solution']:
                                    body += f"💡 **How to fix:**\n{parsed['solution']}\n"
                    else:
                        head = "## ❌ Validation Failed"
                        body = "\n\n😟 **Your algorithm has issues that need to be fixed.**\n\n"
                        
                        if rep.errors:
                            body += "### 🚫 Critical Errors:\n"
                            for i, error in enumerate(rep.errors, 1):
                                parsed = _parse_validation_message(error)
                                body += f"\n#### Error {i}: {parsed['title']}\n"
                                body += f"**What went wrong:** {parsed['description']}\n\n"
                                if parsed['solution']:
                                    body += f"**💡 How to fix it:**\n{parsed['solution']}\n\n"
                                body += "---\n"
                        
                        if rep.warnings:
                            body += "\n### ⚠️ Additional Warnings:\n"
                            for warning in rep.warnings:
                                parsed = _parse_validation_message(warning)
                                body += f"\n**{parsed['title']}**\n"
                                body += f"{parsed['description']}\n\n"
                                if parsed['solution']:
                                    body += f"💡 {parsed['solution']}\n\n"
                    
                    return f"{head}\n{body}"
                
                def _parse_validation_message(msg: str) -> dict:
                    """Parse structured validation messages: CODE|Title|Solution"""
                    parts = msg.split('|')
                    if len(parts) >= 3:
                        return {
                            "code": parts[0].strip(),
                            "title": parts[1].strip(),
                            "description": parts[1].strip(),
                            "solution": parts[2].strip()
                        }
                    elif len(parts) == 2:
                        return {
                            "code": parts[0].strip(),
                            "title": parts[1].strip(),
                            "description": parts[1].strip(),
                            "solution": ""
                        }
                    else:
                        # Fallback for old-style messages
                        return {
                            "code": "UNKNOWN",
                            "title": msg,
                            "description": msg,
                            "solution": ""
                        }

                # Wire up the events
                refresh_btn.click(
                    fn=load_selected_for_compose,
                    inputs=[blueprint_drop, selected_components],
                    outputs=[*stage_selectors.values(), stage_status]
                )
                
                blueprint_drop.change(
                    fn=load_selected_for_compose,
                    inputs=[blueprint_drop, selected_components],
                    outputs=[*stage_selectors.values(), stage_status]
                )
                
                # Auto-load when tab is selected
                compose_tab.select(
                    fn=load_selected_for_compose,
                    inputs=[blueprint_drop, selected_components],
                    outputs=[*stage_selectors.values(), stage_status]
                )

                generate_btn.click(
                    fn=do_generate,
                    inputs=[blueprint_drop, out_name, *stage_selectors.values()],
                    outputs=[code_preview, stage_status]
                )

                validate_btn.click(
                    fn=do_validate,
                    inputs=[blueprint_drop, out_name, *stage_selectors.values()],
                    outputs=[validate_md]
                )

            with gr.Tab("💬 3. Chat Assistant"):
                gr.Markdown("### Ask the AI assistant about cryptography")
                gr.Markdown("Get help with algorithms, components, implementations, and recommendations")
                
                chatbox = gr.Chatbot(label="Cryptography Assistant")
                msg = gr.Textbox(
                    placeholder="Ask about algorithms, components, or your generated cipher…",
                    label="Your Question"
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear_chat = gr.Button("Clear Chat", variant="secondary")
                
                gr.Markdown("""
                **Example questions:**
                - "Show me the AES mix_columns implementation"
                - "What components does Blowfish have?"
                - "Recommend a cipher for high security applications"
                - "Give me full DES implementation"
                """)
                
                def on_send(history, m):
                    if not m.strip():
                        return history, ""
                    history, reply = chat.chat(history or [], m)
                    return history, ""
                
                def clear_history():
                    return []
                
                send.click(on_send, [chatbox, msg], [chatbox, msg])
                msg.submit(on_send, [chatbox, msg], [chatbox, msg])  # Allow Enter key
                clear_chat.click(clear_history, outputs=[chatbox])

        return demo
