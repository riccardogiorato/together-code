/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from 'react';

export const LOADING_MESSAGE = 'Processing...';

/**
 * Custom hook to manage loading messages.
 * @param isActive Whether the loading message should be active.
 * @param isWaiting Whether to show a specific waiting phrase.
 * @returns The current loading message.
 */
export const usePhraseCycler = (isActive: boolean, isWaiting: boolean) => {
  if (isWaiting) {
    return 'Waiting for user confirmation...';
  }

  return LOADING_MESSAGE;
};
