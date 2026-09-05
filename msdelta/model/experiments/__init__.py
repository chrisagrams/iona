"""Architecture experiments, selected at run time with ``--model_class``.

Modules here are deliberately NOT imported by ``msdelta.model``: each declares its own
model classes, and importing them eagerly would be pointless work for a normal run.
None of them call ``register_for_auto_class`` -- doing so would rebind the Hugging Face
auto-classes away from the baseline model.
"""
