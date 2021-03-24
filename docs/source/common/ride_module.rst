.. role:: hidden
    :class: hidden-section

.. _ride_module:

RideModule
===============
A :class:`~RideModule` organizes your PyTorch code into 5 sections

- Computations (init).
- Train loop (training_step)
- Validation loop (validation_step)
- Test loop (test_step)
- Optimizers (configure_optimizers)


    .. code-block:: python

        net = Net()
        trainer = Trainer()
        trainer.fit(net)
