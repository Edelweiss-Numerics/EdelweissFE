#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

from abc import ABC, abstractmethod

import numpy as np

from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.fieldoutput import FieldOutputController
from edelweissfe.utils.plotter import Plotter
from edelweissfe.utils.schema import OptionSchemaProvider


class OutputManagerBase(OptionSchemaProvider, ABC):
    """This is the abstract base class for all output managers.
    User defined output managers must implement the abstract methods.

    Deriving from :class:`~edelweissfe.utils.schema.OptionSchemaProvider` means every output
    manager -- including one supplied by a third-party package via an entry point -- exposes a
    ``schema`` class attribute, so the registry can hand its option schema to the caller alongside
    the class itself. Subclasses that do not define their own option schema simply inherit the
    default of ``None``.

    Parameters
    ----------
    name
        The name of this output manager.
    definitionLines
        The dictionary containing the definition of the output manager.
    model
        A dictionary containing the model tree.
    fieldOutputController
        The field output contoller instance.
    journal
        The journal instance for logging.
    plotter
        The plotter instance for plotting.
    """

    identification = "OutputManagerBase"

    @abstractmethod
    def __init__(
        self,
        name: str,
        definitionLines: dict,
        model: FEModel,
        fieldOutputController: FieldOutputController,
        journal: Journal,
        plotter: Plotter,
    ):
        pass

    def applyOptionsOverride(self, fieldValues: dict) -> None:
        """Apply a partial override of this output manager's own ``schema`` fields.

        The counterpart, on the output manager side, of the name-based ``>>options`` override
        mechanism (``stepactions/options.py``): once that mechanism has resolved an ``>>options,
        name=X, ...`` block to this output manager instance and validated the present keys against
        ``type(self).schema`` via :func:`~edelweissfe.utils.schema.coercePresentOptions`, it calls
        this method with the result to actually apply them.

        Concrete output managers vary in how (or whether) they store overridable runtime options --
        unlike a solver's uniform ``self.options`` dict, there is no single shared storage shape to
        update generically here, so a subclass that wants ``>>options`` support overrides this with
        its own named fields (ordinary polymorphism, not attribute probing -- see
        :class:`OutputManager` in ``ensight.py`` for the one concrete case that needs this today).

        The default here raises rather than silently doing nothing: without an override, a
        ``>>options, name=X, someField=...`` block against ``X`` would otherwise validate cleanly
        against ``type(X).schema`` (``stepactions/options.py``) and then apply no change at all --
        indistinguishable, from the ``.inp`` author's side, from success.

        Parameters
        ----------
        fieldValues
            Maps schema field name to its new, already-coerced value.

        Raises
        ------
        NotImplementedError
            If ``fieldValues`` is non-empty and the subclass has not overridden this method.
        """
        if fieldValues:
            raise NotImplementedError(
                f"{type(self).__name__} does not support '>>options' overrides for "
                f"{sorted(fieldValues)} -- applyOptionsOverride is not implemented for this output manager."
            )

    @abstractmethod
    def initializeJob(self):
        """Initalize the output manager at the beginning of a step.

        Parameters
        ----------
        """

    @abstractmethod
    def initializeStep(self, step: dict):
        """Initalize the output manager at the beginning of a step.

        Parameters
        ----------
        step
            A dictionary containing the step definition.
        """

    @abstractmethod
    def finalizeIncrement(self, timeStep: TimeStep, **kwargs):
        """Finalize the output at the end of a time increment.

        Parameters
        ----------
        U
            The initial solution vector.
        P
            The initial reaction vector.
        timeStep
            The time step.
        **kwargs
            Keyword arguments.
        """

    @abstractmethod
    def finalizeFailedIncrement(self, **kwargs):
        """Finalize the output at the end of a time increment.

        Parameters
        ----------
        **kwargs
            Keyword arguments.
        """

    @abstractmethod
    def finalizeStep(
        self,
    ):
        """Finalize the output the end of a step."""

    @abstractmethod
    def finalizeJob(
        self,
    ):
        """Finalize the output at the end of a job.

        Parameters
        ----------
        U
            The final solution vector.
        P
            The final reaction vector.
        """

    def getRestartData(self) -> dict[str, np.ndarray] | None:
        """Return this output manager's own sequence/bookkeeping state needed to continue its
        output correctly after a resume (e.g. Ensight's per-time-set history of already-written
        time values, which its file/frame numbering is derived from), to be bundled into a restart
        checkpoint by the restart-writing output manager (``outputmanagers/restart.py``), or
        ``None`` if this output manager has nothing that would otherwise go stale on resume.

        The default implementation returns ``None`` -- correct for most output managers, whose own
        state is either trivially re-derivable (e.g. a status file that just keeps appending) or
        irrelevant to correctness (only Ensight's transient sequence numbering silently corrupts
        without this, since a resumed run's fresh instance would otherwise restart file/frame
        numbering from zero, orphaning-and/or corrupting the pre-resume portion of the sequence).

        Returns
        -------
        dict[str, np.ndarray] | None
            A flat mapping of array name to array, or ``None``.
        """

        return None

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore this output manager's own sequence/bookkeeping state from a restart checkpoint.

        Parameters
        ----------
        data
            The mapping previously returned by :meth:`getRestartData`.
        """

        raise NotImplementedError("This output manager does not carry restartable sequence state.")
