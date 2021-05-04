# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except
# in compliance with the License. A copy of the License is located at
#
# https://aws.amazon.com/apache-2-0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Shared utility helper functions"""
import os
from io import BytesIO

EXIF_ORIENTATION = 274  # Magic numbers from http://www.exiv2.org/tags.html

def random_hex_bytes(n_bytes):
    """Create a hex encoded string of random bytes"""
    return os.urandom(n_bytes).hex()
