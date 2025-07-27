/*
  +----------------------------------------------------------------------+
  | ort                                                                  |
  +----------------------------------------------------------------------+
  | Copyright (c) Joe Watkins 2025                                       |
  +----------------------------------------------------------------------+
  | This source file is subject to version 3.01 of the PHP license,      |
  | that is bundled with this package in the file LICENSE, and is        |
  | available through the world-wide-web at the following url:           |
  | http://www.php.net/license/3_01.txt                                  |
  | If you did not receive a copy of the PHP license and are unable to   |
  | obtain it through the world-wide-web, please send a note to          |
  | license@php.net so we can mail you a copy immediately.               |
  +----------------------------------------------------------------------+
  | Author: krakjoe                                                      |
  +----------------------------------------------------------------------+
 */

/**
 * Shall be true when MINIT yields SUCCESS
 */
Module.ready = false;

/**
 * Shall startup (MINIT) em
 * @returns bool
 */
Module.startup = function() {
    return Module.ready = 
        !Module.ccall(
            'em_startup', 
            'number');
};

/**
 * Shall invoke code
 * 
 * Where input is HTMLTextAreaElement:
 *  input will be taken from input.value
 * Where input is Function:
 *  input will be invoked thus: function()
 * 
 * Note: input Function should return an object with value and length fields
 * 
 * Where output is HTMLElement:
 *  text will be inserted into output.textContent, and returned
 * Where output is Function:
 *  output will be invoked thus: output(text), and the result returned
 * Where output is undefined:
 *  text will be returned
 * 
 * @param {(HTMLTextAreaElement|Function|string)} input 
 * @param {(HTMLElement|Function|undefined)} output 
 * @returns string
 */
Module.invoke = function(input, output = undefined) {
    let code = null;

    if (typeof HTMLTextAreaElement !== 'undefined' &&
        input instanceof HTMLTextAreaElement) {
        code = {
            value:  input.value,
            length: Module.lengthBytesUTF8(
                input.value),
        };
    } else if (typeof input === 'function') {
        code = input();

        if (typeof code !== 'object') {
            throw new TypeError(
                "Unexpected result from input(), " +
                "expected an object");
        }

        if (typeof code.value  !== 'string' ||
            typeof code.length !== 'number') {
            throw new TypeError(
                "Unexpected result from input(), " +
                "expected {string value, number length}");
        }
    } else if (typeof input == 'string') {
        code = {
            value:  input,
            length: Module.lengthBytesUTF8(
                input),
        };
    } else {
        throw new TypeError(
            "Unexpected input " +
            "expected HTMLTextAreaElement|Function|string");
    }

    if (!code.value || !code.length) {
        throw new Error("Unexpected input, zero length");
    }

    // Fire start event
    Module.dispatchEvent(new CustomEvent('invoke.begin', { 
        detail: { 
            "input": input,
            "output": output 
        } 
    }));

    // run the code, getting it's address and length in return
    let result = {
        address: Module.ccall(
            'em_run_string',
            'number',
            ['string', 'number'],
            [code.value, code.length]),
        length: Module.ccall(
            'em_run_length', 'number')
    };

    // check for errors
    if (result.address < 0) {
        // Fire error event
        Module.dispatchEvent(new CustomEvent('invoke.error', { 
            detail: { 
                "input": input,
                "output": output,
                "result": result }
        }));

        // we don't need to care about freeing, nothing was allocated
        throw new Error("Unexpected result, execution failed");
    }

    // ensure there's stuff to read on the heap
    if (!result.length) {
        // Fire error event
        Module.dispatchEvent(new CustomEvent('invoke.error', { 
            detail: { 
                "input": input,
                "output": output,
                "result": result }
        }));

        // we don't need to care about freeing, nothing was allocated
        throw new Error("Unexpected result, no output");
    }

    let heap = null;
    let text = null;

    try {
        heap = new Uint8Array // into a buffer that js owns
        (                 
            Module.HEAPU8.subarray(              // from the heap that em alloc'd
                result.address,                  // start at address returned by run
                result.address + result.length)  // up to length returned by run (handle nulls)
        ); 

        // decoded from js owned buffer
        text = Module.decoder.decode(heap);
    } catch (exception) {
        // Fire exception event
        Module.dispatchEvent(new CustomEvent('invoke.exception', { 
            detail: { 
                "input": input,
                "output": output,
                "result": result,
                "exception": exception }
        }));

        throw exception;
    } finally {
        // release the buffer that em alloc'd
        Module.ccall('em_run_free');
    }

   // Fire end event
    Module.dispatchEvent(new CustomEvent('invoke.end', { 
        detail: { 
            "input":  input,
            "output": output,
            "text":   text }
    }));

    if (typeof output === 'undefined') {
        return text;
    } else if (typeof HTMLElement !== 'undefined' &&
        output instanceof HTMLElement) {
        return output.textContent = text;
    } else if (typeof output === 'function') {
        return output(text);
    }

    throw new TypeError(
        "Unexpected output type, " +
        "expected HTMLElement|Function|undefined");
};

/**
 * Shall shutdown (MSHUTDOWN) em
 * @returns void
 */
Module.shutdown = function() {
    Module.ccall(
        'em_shutdown');
};

Module['onRuntimeInitialized'] = function() {
    Module.startup();

    if (typeof window !== 'undefined') {
        window.addEventListener('unload', function() {
            if (typeof Module !== 'undefined' && Module.shutdown) {
                Module.shutdown();
            }
        });
    }

    Module.decoder = new TextDecoder();
    Module.events = new EventTarget();
    Module.addEventListener = (type, fn) =>
        Module.events.addEventListener(type, fn);
    Module.dispatchEvent = (event) =>
        Module.events.dispatchEvent(event);
};