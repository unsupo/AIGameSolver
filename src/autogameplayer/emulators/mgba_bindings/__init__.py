# Copyright (c) 2013-2017 Jeffrey Pfau
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from ._pylib import ffi, lib  # pylint: disable=no-name-in-module

class Git:
    commit = None
    commitShort = None
    branch = None
    revision = None

def create_callback(struct_name, cb_name, func_name=None):
    func_name = func_name or "_py{}{}".format(struct_name, cb_name[0].upper() + cb_name[1:])
    full_struct = "struct {}*".format(struct_name)

    def callback(handle, *args):
        handle = ffi.cast(full_struct, handle)
        return getattr(ffi.from_handle(handle.pyobj), cb_name)(*args)

    return ffi.def_extern(name=func_name)(callback)

__version__ = "0.10.5"
